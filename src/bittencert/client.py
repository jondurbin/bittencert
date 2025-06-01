"""
aiohttp client helpers for use with bittencert
"""

import ssl
import socket
from abc import ABC, abstractmethod
from typing import Optional, Any
from datetime import datetime, timezone
from dataclasses import dataclass
from aiohttp import TCPConnector, ClientSession
from cryptography import x509
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.backends import default_backend
from collections import OrderedDict, namedtuple
import bittencert

CacheInfo = namedtuple("CacheInfo", ["hits", "misses", "maxsize", "currsize"])


@dataclass
class CertificateEntry:
    """
    Container for a verified certificate with metadata.
    """

    certificate: x509.Certificate
    pem_bytes: bytes
    ss58_address: str
    verified_at: datetime
    expires_at: datetime

    @property
    def is_expired(self) -> bool:
        if self.expires_at.tzinfo:
            return datetime.now(timezone.utc) > self.expires_at
        return datetime.now(timezone.utc).replace(tzinfo=None) > self.expires_at

    def to_dict(self) -> dict[str, Any]:
        return {
            "pem_bytes": self.pem_bytes.decode("utf-8"),
            "ss58_address": self.ss58_address,
            "verified_at": self.verified_at.isoformat(),
            "expires_at": self.expires_at.isoformat(),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "CertificateEntry":
        pem_bytes = data["pem_bytes"].encode("utf-8")
        cert = x509.load_pem_x509_certificate(pem_bytes, default_backend())
        return cls(
            certificate=cert,
            pem_bytes=pem_bytes,
            ss58_address=data["ss58_address"],
            verified_at=datetime.fromisoformat(data["verified_at"]),
            expires_at=datetime.fromisoformat(data["expires_at"]),
        )


class CertificateStore(ABC):
    """
    Abstract base class for certificate storage backends.
    """

    @abstractmethod
    async def get(self, host: str, port: int, ss58_address: str) -> Optional[CertificateEntry]:
        pass

    @abstractmethod
    async def put(self, host: str, port: int, entry: CertificateEntry) -> None:
        pass

    @abstractmethod
    async def delete(self, host: str, port: int, ss58_address: str) -> None:
        pass

    @abstractmethod
    async def clear(self) -> None:
        pass

    def make_key(self, host: str, port: int, ss58_address: str) -> str:
        return f"{host}:{port}:{ss58_address}"

    async def get_valid(
        self, host: str, port: int, ss58_address: str
    ) -> Optional[CertificateEntry]:
        entry = await self.get(host, port, ss58_address)
        if entry and not entry.is_expired:
            return entry
        elif entry:
            await self.delete(host, port, ss58_address)
        return None


class LRUCertificateStore(CertificateStore):
    """
    In-memory LRU cache implementation of certificate store.
    """

    def __init__(self, maxsize: int = 128):
        self.maxsize = maxsize
        self._cache = OrderedDict()
        self._hits = 0
        self._misses = 0

    async def get(self, host: str, port: int, ss58_address: str) -> Optional[CertificateEntry]:
        key = self.make_key(host, port, ss58_address)
        if key in self._cache:
            self._cache.move_to_end(key)
            self._hits += 1
            return self._cache[key]
        else:
            self._misses += 1
            return None

    async def put(self, host: str, port: int, entry: CertificateEntry) -> None:
        key = self.make_key(host, port, entry.ss58_address)
        if key in self._cache:
            self._cache.move_to_end(key)
        self._cache[key] = entry
        if len(self._cache) > self.maxsize:
            self._cache.popitem(last=False)

    async def delete(self, host: str, port: int, ss58_address: str) -> None:
        key = self.make_key(host, port, ss58_address)
        if key in self._cache:
            del self._cache[key]

    async def clear(self) -> None:
        self._cache.clear()
        self._hits = 0
        self._misses = 0

    def cache_info(self):
        return CacheInfo(
            hits=self._hits, misses=self._misses, maxsize=self.maxsize, currsize=len(self._cache)
        )


class BittencertConnector(TCPConnector):
    """
    Custom aiohttp connector that verifies Bittensor certificates.
    """

    def __init__(
        self,
        ss58_address: str,
        cert_store: Optional[CertificateStore] = None,
        verify_hostname: Optional[bool] = True,
        connect_timeout: Optional[int] = 10.0,
        *args,
        **kwargs,
    ):
        self.ss58_address = ss58_address
        self.cert_store = cert_store or LRUCertificateStore()
        self.verify_hostname = verify_hostname
        self.connect_timeout = connect_timeout
        super().__init__(*args, ssl=False, **kwargs)

    async def _verify_and_store_certificate(self, host: str, port: int) -> CertificateEntry:
        cert_der = await self._fetch_server_certificate(host, port)
        cert = x509.load_der_x509_certificate(cert_der)
        if not bittencert.verify(cert, ss58_address=self.ss58_address, verify_hostname=host):
            raise Exception(f"Certificate verification failed for {host}")
        cert_pem = cert.public_bytes(serialization.Encoding.PEM)
        entry = CertificateEntry(
            certificate=cert,
            pem_bytes=cert_pem,
            ss58_address=self.ss58_address,
            verified_at=datetime.now(timezone.utc),
            expires_at=cert.not_valid_after_utc,
        )
        await self.cert_store.put(host, port, entry)
        return entry

    async def _create_connection(self, req, traces, timeout):
        host = req.url.host
        port = req.url.port or 443
        try:
            if (entry := await self.cert_store.get_valid(host, port, self.ss58_address)) is None:
                entry = await self._verify_and_store_certificate(host, port)
            ssl_context = ssl.create_default_context()
            ssl_context.load_verify_locations(cadata=entry.pem_bytes.decode("utf-8"))
            self._ssl = ssl_context
        except Exception as e:
            raise Exception(f"Bittensor certificate verification failed: {e}")
        return await super()._create_connection(req, traces, timeout)

    async def _fetch_server_certificate(self, host: str, port: int) -> bytes:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(self.connect_timeout)
        context = ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
        context.check_hostname = False
        context.verify_mode = ssl.CERT_NONE
        with context.wrap_socket(sock, server_hostname=host) as ssock:
            ssock.connect((host, port))
            cert_der = ssock.getpeercert(binary_form=True)
        return cert_der


class BittencertSession(ClientSession):
    """
    Custom aiohttp session to make use of our verification.
    """

    def __init__(self, ss58_address: str, cert_store: Optional[CertificateStore] = None, **kwargs):
        self.cert_store = cert_store or LRUCertificateStore()
        self._connector = BittencertConnector(ss58_address, self.cert_store)
        super().__init__(connector=self._connector, **kwargs)

    async def clear_cert_store(self):
        await self.cert_store.clear()
