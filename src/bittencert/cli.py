import json
import typer
import asyncio
from datetime import datetime, timedelta, timezone
from bittencert.generate import generate
from bittencert.client import BittencertSession
from cryptography.hazmat.primitives import serialization
from substrateinterface import Keypair


def generate_cert(
    hotkey_path: str = typer.Option(
        ...,
        "--hotkey-path",
        help="Path to the hotkey to generate a certificate from.",
    ),
    cn: str = typer.Option(..., "--cn", help="Certificate common name CN"),
    ou: str = typer.Option(..., "--ou", help="Certificate organization unit name OU"),
    serial: int = typer.Option(None, "--serial", help="Certificate serial number"),
    days: int = typer.Option(30, "--days", help="Certificate validity, in days"),
    sans: list[str] = typer.Option(None, "--sans", help="Subject alternate names SANs"),
    key_path: str = typer.Option("key.pem", "--key-path", help="Path to save the private key to."),
    cert_path: str = typer.Option(
        "cert.pem", "--cert-path", help="Path to save the certificate file to."
    ),
):
    with open(hotkey_path, "r") as infile:
        secret_seed = json.loads(infile.read())["secretSeed"]
    keypair = Keypair.create_from_seed(secret_seed[2:])
    from_date = datetime.now(timezone.utc).replace(microsecond=0, second=0, tzinfo=timezone.utc)
    key, cert, signature = generate(
        keypair,
        cn=cn,
        ou=ou,
        serial=serial,
        from_date=from_date,
        to_date=from_date + timedelta(days=days),
    )
    cert_pem = cert.public_bytes(serialization.Encoding.PEM)
    key_pem = key.private_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PrivateFormat.PKCS8,
        encryption_algorithm=serialization.NoEncryption(),
    )
    with open(key_path, "wb") as outfile:
        outfile.write(key_pem)
    with open(cert_path, "wb") as outfile:
        outfile.write(cert_pem)
    print(
        f"Successfully generated certificate with signature: {signature}\n  {key_path=}\n  {cert_path=}"
    )


def verify_cert(
    ss58: str = typer.Option(..., "--ss58", help="ss58 address of the entity running the server"),
    url: str = typer.Option(..., "--url", help="URL to fetch"),
):
    async def _verify_cert():
        async with BittencertSession(ss58) as session:
            async with session.get(url):
                print("Successfully verified certificate signature/validity!")

    return asyncio.run(_verify_cert())


app = typer.Typer(no_args_is_help=True)

app.command(name="generate", help="Generate a new cert from a hotkey file.")(generate_cert)
app.command(name="verify", help="Verify a certificate on a running server.")(verify_cert)

if __name__ == "__main__":
    app()
