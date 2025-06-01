"""
Tests for bittencert.
"""

import datetime
import uuid
from typing import Tuple
import pytest
from cryptography import x509
from cryptography.hazmat.primitives import serialization, hashes
from cryptography.hazmat.primitives.asymmetric import ec
from substrateinterface import Keypair
import uvicorn
from fastapi import FastAPI
from multiprocessing import Process
import time
from bittencert import generate, verify
from bittencert.client import (
    CertificateEntry,
    LRUCertificateStore,
    BittencertSession,
)


@pytest.fixture
def test_keypair():
    return Keypair.create_from_mnemonic(
        "bottom drive obey lake curtain smoke basket hold race lonely fit walk", ss58_format=42
    )


@pytest.fixture
def test_keypair_2():
    return Keypair.create_from_mnemonic(
        "ancient young hurt bone shuffle deposit congress normal crack six boost despair",
        ss58_format=42,
    )


@pytest.fixture
def test_cert_and_key(
    test_keypair,
) -> Tuple[
    x509.Certificate,
    ec.EllipticCurvePrivateKey,
]:
    private_key, cert, _ = generate(test_keypair, cn="test.example.com")
    return cert, private_key


class TestGenerate:
    def test_generate_default_values(self, test_keypair):
        private_key, cert, signature = generate(test_keypair)
        assert isinstance(private_key, ec.EllipticCurvePrivateKey)
        assert isinstance(cert, x509.Certificate)
        assert isinstance(signature, str)
        assert cert.subject
        cn = cert.subject.get_attributes_for_oid(x509.NameOID.COMMON_NAME)[0].value
        assert cn.startswith("bt-node-")
        ou = cert.subject.get_attributes_for_oid(x509.NameOID.ORGANIZATIONAL_UNIT_NAME)[0].value
        assert uuid.UUID(ou)
        o = cert.subject.get_attributes_for_oid(x509.NameOID.ORGANIZATION_NAME)[0].value
        assert len(o) == 128

    def test_generate_custom_values(self, test_keypair):
        from_date = datetime.datetime.now(datetime.timezone.utc).replace(second=0, microsecond=0)
        to_date = from_date + datetime.timedelta(days=90)
        private_key, cert, signature = generate(
            test_keypair,
            cn="custom.example.com",
            ou="custom-ou-value",
            serial=12345,
            from_date=from_date,
            to_date=to_date,
            sans=["alt1.example.com", "alt2.example.com"],
        )
        cn = cert.subject.get_attributes_for_oid(x509.NameOID.COMMON_NAME)[0].value
        assert cn == "custom.example.com"
        ou = cert.subject.get_attributes_for_oid(x509.NameOID.ORGANIZATIONAL_UNIT_NAME)[0].value
        assert ou == "custom-ou-value"
        assert cert.serial_number == 12345
        assert cert.not_valid_before_utc == from_date
        assert cert.not_valid_after_utc == to_date
        san_ext = cert.extensions.get_extension_for_class(x509.SubjectAlternativeName).value
        dns_names = [san.value for san in san_ext]
        assert "alt1.example.com" in dns_names
        assert "alt2.example.com" in dns_names

    def test_generate_reproducible_ou(self, test_keypair):
        _, cert1, _ = generate(test_keypair)
        _, cert2, _ = generate(test_keypair)
        ou1 = cert1.subject.get_attributes_for_oid(x509.NameOID.ORGANIZATIONAL_UNIT_NAME)[0].value
        ou2 = cert2.subject.get_attributes_for_oid(x509.NameOID.ORGANIZATIONAL_UNIT_NAME)[0].value
        assert ou1 == ou2


class TestVerify:
    def test_verify_valid_certificate(self, test_keypair):
        _, cert, _ = generate(test_keypair)
        assert verify(cert, ss58_address=test_keypair.ss58_address) is True

    def test_verify_invalid_signature(self, test_keypair, test_keypair_2):
        _, cert, _ = generate(test_keypair)
        assert verify(cert, ss58_address=test_keypair_2.ss58_address) is False

    def test_verify_missing_fields(self):
        private_key = ec.generate_private_key(ec.SECP256R1())
        builder = x509.CertificateBuilder()
        builder = builder.subject_name(
            x509.Name(
                [
                    x509.NameAttribute(x509.NameOID.COMMON_NAME, "test.com"),
                ]
            )
        )
        builder = builder.issuer_name(builder._subject_name)
        builder = builder.public_key(private_key.public_key())
        builder = builder.serial_number(x509.random_serial_number())
        builder = builder.not_valid_before(datetime.datetime.now(datetime.timezone.utc))
        builder = builder.not_valid_after(
            datetime.datetime.now(datetime.timezone.utc) + datetime.timedelta(days=30)
        )
        cert = builder.sign(private_key, hashes.SHA256())
        assert not verify(cert, ss58_address="5GrwvaEF5zXb26Fz9rcQpDWS57CtERHpNehXCPcNoHGKutQY")

    def test_verify_tampered_certificate(self, test_keypair):
        _, cert, _ = generate(test_keypair, cn="original.com")
        cert_data = cert.public_bytes(serialization.Encoding.DER)
        cert = x509.load_der_x509_certificate(cert_data)
        assert verify(cert, ss58_address=test_keypair.ss58_address) is True


class TestCertificateEntry:
    def test_certificate_entry_creation(self, test_cert_and_key, test_keypair):
        cert, _ = test_cert_and_key
        pem_bytes = cert.public_bytes(serialization.Encoding.PEM)
        entry = CertificateEntry(
            certificate=cert,
            pem_bytes=pem_bytes,
            ss58_address=test_keypair.ss58_address,
            verified_at=datetime.datetime.utcnow(),
            expires_at=cert.not_valid_after_utc,
        )
        assert entry.certificate == cert
        assert entry.pem_bytes == pem_bytes
        assert entry.ss58_address == test_keypair.ss58_address
        assert not entry.is_expired

    def test_certificate_entry_expired(self, test_cert_and_key, test_keypair):
        cert, _ = test_cert_and_key
        pem_bytes = cert.public_bytes(serialization.Encoding.PEM)
        entry = CertificateEntry(
            certificate=cert,
            pem_bytes=pem_bytes,
            ss58_address=test_keypair.ss58_address,
            verified_at=datetime.datetime.utcnow(),
            expires_at=datetime.datetime.utcnow() - datetime.timedelta(days=1),
        )
        assert entry.is_expired

    def test_certificate_entry_serialization(self, test_cert_and_key, test_keypair):
        cert, _ = test_cert_and_key
        pem_bytes = cert.public_bytes(serialization.Encoding.PEM)
        original = CertificateEntry(
            certificate=cert,
            pem_bytes=pem_bytes,
            ss58_address=test_keypair.ss58_address,
            verified_at=datetime.datetime.utcnow(),
            expires_at=cert.not_valid_after_utc,
        )
        data = original.to_dict()
        restored = CertificateEntry.from_dict(data)
        assert restored.pem_bytes == original.pem_bytes
        assert restored.ss58_address == original.ss58_address
        assert restored.verified_at == original.verified_at
        assert restored.expires_at == original.expires_at


class TestLRUCertificateStore:
    @pytest.mark.asyncio
    async def test_store_and_retrieve(self, test_cert_and_key, test_keypair):
        cert, _ = test_cert_and_key
        pem_bytes = cert.public_bytes(serialization.Encoding.PEM)
        store = LRUCertificateStore(maxsize=10)
        entry = CertificateEntry(
            certificate=cert,
            pem_bytes=pem_bytes,
            ss58_address=test_keypair.ss58_address,
            verified_at=datetime.datetime.utcnow(),
            expires_at=cert.not_valid_after_utc,
        )
        await store.put("test.com", 443, entry)
        retrieved = await store.get("test.com", 443, test_keypair.ss58_address)
        assert retrieved is not None
        assert retrieved.ss58_address == entry.ss58_address
        missing = await store.get("other.com", 443, test_keypair.ss58_address)
        assert missing is None

    @pytest.mark.asyncio
    async def test_store_delete(self, test_cert_and_key, test_keypair):
        cert, _ = test_cert_and_key
        pem_bytes = cert.public_bytes(serialization.Encoding.PEM)
        store = LRUCertificateStore()
        entry = CertificateEntry(
            certificate=cert,
            pem_bytes=pem_bytes,
            ss58_address=test_keypair.ss58_address,
            verified_at=datetime.datetime.utcnow(),
            expires_at=cert.not_valid_after_utc,
        )
        await store.put("test.com", 443, entry)
        await store.delete("test.com", 443, test_keypair.ss58_address)
        retrieved = await store.get("test.com", 443, test_keypair.ss58_address)
        assert retrieved is None

    @pytest.mark.asyncio
    async def test_store_clear(self, test_cert_and_key, test_keypair):
        cert, _ = test_cert_and_key
        pem_bytes = cert.public_bytes(serialization.Encoding.PEM)
        store = LRUCertificateStore()
        entry = CertificateEntry(
            certificate=cert,
            pem_bytes=pem_bytes,
            ss58_address=test_keypair.ss58_address,
            verified_at=datetime.datetime.utcnow(),
            expires_at=cert.not_valid_after_utc,
        )
        await store.put("test1.com", 443, entry)
        await store.put("test2.com", 443, entry)
        await store.clear()
        assert await store.get("test1.com", 443, test_keypair.ss58_address) is None
        assert await store.get("test2.com", 443, test_keypair.ss58_address) is None

    @pytest.mark.asyncio
    async def test_store_get_valid_expires(self, test_cert_and_key, test_keypair):
        cert, _ = test_cert_and_key
        pem_bytes = cert.public_bytes(serialization.Encoding.PEM)
        store = LRUCertificateStore()
        expired_entry = CertificateEntry(
            certificate=cert,
            pem_bytes=pem_bytes,
            ss58_address=test_keypair.ss58_address,
            verified_at=datetime.datetime.utcnow(),
            expires_at=datetime.datetime.utcnow() - datetime.timedelta(days=1),
        )
        await store.put("test.com", 443, expired_entry)
        result = await store.get_valid("test.com", 443, test_keypair.ss58_address)
        assert result is None
        assert await store.get("test.com", 443, test_keypair.ss58_address) is None


def create_test_app(keypair: Keypair):
    app = FastAPI()

    @app.get("/")
    async def root():
        return {"message": "hi", "address": keypair.ss58_address}

    @app.get("/data")
    async def get_data():
        return {"data": list(range(10)), "timestamp": datetime.datetime.utcnow().isoformat()}

    return app


def run_test_server(keypair: Keypair, key_file: str, cert_file: str, port: int):
    app = create_test_app(keypair)
    uvicorn.run(
        app,
        host="127.0.0.1",
        port=port,
        ssl_keyfile=key_file,
        ssl_certfile=cert_file,
        log_level="error",
    )


class TestIntegration:
    @pytest.fixture
    def test_server_port(self):
        import socket

        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind(("", 0))
            return s.getsockname()[1]

    @pytest.fixture
    def test_server(self, test_keypair, test_server_port, tmp_path):
        private_key, cert, _ = generate(
            test_keypair, cn="localhost", sans=["localhost", "127.0.0.1"]
        )
        cert_file = tmp_path / "cert.pem"
        key_file = tmp_path / "key.pem"
        cert_file.write_bytes(cert.public_bytes(serialization.Encoding.PEM))
        key_file.write_bytes(
            private_key.private_bytes(
                encoding=serialization.Encoding.PEM,
                format=serialization.PrivateFormat.PKCS8,
                encryption_algorithm=serialization.NoEncryption(),
            )
        )
        process = Process(
            target=run_test_server,
            args=(test_keypair, str(key_file), str(cert_file), test_server_port),
        )
        process.start()
        time.sleep(1)
        yield f"https://localhost:{test_server_port}", test_keypair.ss58_address
        process.terminate()
        process.join()

    @pytest.mark.asyncio
    async def test_bittencert_session(self, test_server):
        url, ss58_address = test_server
        async with BittencertSession(ss58_address=ss58_address) as session:
            async with session.get(url) as response:
                assert response.status == 200
                data = await response.json()
                assert data["address"] == ss58_address
            async with session.get(f"{url}/data") as response:
                assert response.status == 200
                data = await response.json()
                assert "data" in data

    @pytest.mark.asyncio
    async def test_bittencert_session_wrong_address(self, test_server, test_keypair_2):
        url, _ = test_server
        wrong_address = test_keypair_2.ss58_address
        async with BittencertSession(ss58_address=wrong_address) as session:
            with pytest.raises(Exception) as exc_info:
                async with session.get(url):
                    pass
            assert "verification failed" in str(exc_info.value).lower()

    @pytest.mark.asyncio
    async def test_bittencert_connector_caching(self, test_server):
        url, ss58_address = test_server
        store = LRUCertificateStore()
        async with BittencertSession(ss58_address=ss58_address, cert_store=store) as session:
            async with session.get(url) as response:
                assert response.status == 200
            cache_info = store.cache_info()
            assert cache_info.hits == 0

            async with session.get(url) as response:
                assert response.status == 200

            cache_info = store.cache_info()
            assert cache_info.hits > 0

    @pytest.mark.asyncio
    async def test_multiple_servers_different_certs(self, test_keypair, test_keypair_2, tmp_path):
        servers = []
        for i, keypair in enumerate([test_keypair, test_keypair_2]):
            import socket

            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.bind(("", 0))
                port = s.getsockname()[1]
            private_key, cert, _ = generate(
                keypair, cn="localhost", sans=["localhost", "127.0.0.1"]
            )
            cert_file = tmp_path / f"cert{i}.pem"
            key_file = tmp_path / f"key{i}.pem"
            cert_file.write_bytes(cert.public_bytes(serialization.Encoding.PEM))
            key_file.write_bytes(
                private_key.private_bytes(
                    encoding=serialization.Encoding.PEM,
                    format=serialization.PrivateFormat.PKCS8,
                    encryption_algorithm=serialization.NoEncryption(),
                )
            )
            process = Process(
                target=run_test_server, args=(keypair, str(key_file), str(cert_file), port)
            )
            process.start()
            servers.append((process, port, keypair.ss58_address))
        time.sleep(1)

        try:
            for process, port, address in servers:
                url = f"https://localhost:{port}"
                async with BittencertSession(ss58_address=address) as session:
                    async with session.get(url) as response:
                        assert response.status == 200
                other_address = (
                    test_keypair.ss58_address
                    if address == test_keypair_2.ss58_address
                    else test_keypair_2.ss58_address
                )
                async with BittencertSession(ss58_address=other_address) as session:
                    with pytest.raises(Exception):
                        async with session.get(url) as response:
                            pass

        finally:
            for process, _, _ in servers:
                process.terminate()
                process.join()


class TestCertificateFiles:
    def test_save_and_load_certificate(self, test_keypair, tmp_path):
        private_key, cert, _ = generate(test_keypair)
        cert_file = tmp_path / "server.crt"
        key_file = tmp_path / "server.key"
        cert_pem = cert.public_bytes(serialization.Encoding.PEM)
        key_pem = private_key.private_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PrivateFormat.PKCS8,
            encryption_algorithm=serialization.NoEncryption(),
        )
        cert_file.write_bytes(cert_pem)
        key_file.write_bytes(key_pem)
        loaded_cert = x509.load_pem_x509_certificate(cert_file.read_bytes())
        assert verify(loaded_cert, ss58_address=test_keypair.ss58_address)
        loaded_key = serialization.load_pem_private_key(key_file.read_bytes(), password=None)
        assert isinstance(loaded_key, ec.EllipticCurvePrivateKey)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
