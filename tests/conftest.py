import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from blackbook.storage import Database, Source, Document, Chunk  # noqa: E402
from blackbook.storage.database import sha256_text  # noqa: E402


@pytest.fixture()
def db(tmp_path):
    """An isolated, in-tmp-path database."""
    database = Database(tmp_path / "test.db")
    yield database
    database.close()


@pytest.fixture()
def seeded_db(db):
    """A database seeded with a couple of sources/documents/chunks."""
    with db.session():
        db.upsert_source(Source(source_id="hacktricks", name="HackTricks", authority="trusted"))
        db.upsert_source(Source(source_id="0xdf", name="0xdf", authority="trusted"))
        d1 = db.upsert_document(
            Document(
                source_id="hacktricks",
                external_id="ad/kerberoasting.md",
                title="Kerberoasting",
                url="https://book.hacktricks.xyz/ad/kerberoasting",
                content_hash=sha256_text("kerberoasting doc"),
                categories=["active-directory", "windows"],
            )
        )
        d2 = db.upsert_document(
            Document(
                source_id="0xdf",
                external_id="htb-forest",
                title="HTB: Forest",
                url="https://0xdf.gitlab.io/2019/12/21/htb-forest.html",
                content_hash=sha256_text("forest doc"),
                categories=["htb", "windows"],
            )
        )
        db.replace_chunks(
            d1,
            [
                Chunk(
                    doc_id=d1, ordinal=0,
                    text="Kerberoasting requests service tickets for SPN accounts and cracks them offline.",
                    section_path=["Active Directory", "Kerberos", "Kerberoasting"],
                    token_estimate=15, content_hash=sha256_text("c1"),
                ),
                Chunk(
                    doc_id=d1, ordinal=1,
                    text="Use GetUserSPNs.py from Impacket to request RC4 service tickets.",
                    section_path=["Active Directory", "Kerberos", "Kerberoasting", "Tools"],
                    token_estimate=11, content_hash=sha256_text("c2"),
                ),
            ],
        )
        db.replace_chunks(
            d2,
            [
                Chunk(
                    doc_id=d2, ordinal=0,
                    text="On Forest I kerberoast a service account after AS-REP roasting another user.",
                    section_path=["HTB: Forest", "Shell as svc-alfresco"],
                    token_estimate=14, content_hash=sha256_text("c3"),
                )
            ],
        )
    return db
