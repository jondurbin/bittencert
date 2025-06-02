"""
Certificate generation.
"""

import datetime
import uuid
import ipaddress
from typing import Tuple
from cryptography import x509
from cryptography.x509.oid import NameOID
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.asymmetric import ec
from substrateinterface import Keypair


def generate(
    keypair: Keypair,
    cn: str = None,
    ou: str = None,
    serial: str = None,
    from_date: datetime.datetime = None,
    to_date: datetime.datetime = None,
    sans: list[str] = [],
) -> Tuple[ec.EllipticCurvePrivateKey, x509.Certificate, bytes]:
    """
    Create an Ed25519 TLS certificate signed by a Bittensor keypair.
    """
    private_key = ec.generate_private_key(ec.SECP256R1())
    builder = x509.CertificateBuilder()

    # Default values for the cert.
    if not serial:
        serial = x509.random_serial_number()
    if not cn:
        cn = f"bt-node-{hex(serial)[2:]}"
    if not ou:
        # Rather than use the public key directly, which would allow some web scraper
        # to identify which nodes belong to which miners, we'll use a uuid5 of the
        # public key which adds a tiny bit of obfuscation. Better yet would be to
        # set this to some completely random UUID value that doesn't directly map to
        # miners, but it's a fairly sensible default.
        ou = str(uuid.uuid5(uuid.NAMESPACE_OID, keypair.ss58_address))
    if not from_date:
        from_date = datetime.datetime.now(datetime.timezone.utc)
    if not to_date:
        to_date = datetime.datetime.now(datetime.timezone.utc) + datetime.timedelta(days=30)

    # Adjust cert date precision for signatures.
    from_date = from_date.replace(microsecond=0, second=0, tzinfo=datetime.timezone.utc)
    to_date = to_date.replace(microsecond=0, second=0, tzinfo=datetime.timezone.utc)

    # Create a signature using the keypair to include in the cert.
    cert_string = ":".join(
        [
            str(serial),
            cn,
            ou,
            from_date.isoformat(),
            to_date.isoformat(),
        ]
    )
    signature = keypair.sign(cert_string).hex()

    # Subject with Bittensor identity
    builder = builder.subject_name(
        x509.Name(
            [
                x509.NameAttribute(NameOID.COMMON_NAME, cn),
                x509.NameAttribute(NameOID.ORGANIZATIONAL_UNIT_NAME, ou),
                x509.NameAttribute(NameOID.ORGANIZATION_NAME, signature),
            ]
        )
    )

    # More attributes...
    builder = (
        builder.issuer_name(builder._subject_name)
        .public_key(private_key.public_key())
        .serial_number(serial)
        .not_valid_before(from_date)
        .not_valid_after(to_date)
    )

    # Add subject alternative names.
    if not sans:
        sans = [cn]
    san_list = []
    for san in sans:
        try:
            ip = ipaddress.ip_address(san)
            san_list.append(x509.IPAddress(ip))
        except ValueError:
            san_list.append(x509.DNSName(san))
    builder = builder.add_extension(
        x509.SubjectAlternativeName(san_list),
        critical=False,
    )

    # Sign and done.
    cert = builder.sign(private_key, hashes.SHA256())
    return private_key, cert, signature
