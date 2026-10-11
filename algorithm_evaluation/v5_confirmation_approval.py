"""Owner-installed data fields for the CTBF v5 confirmation.

The values deliberately remain unset while the protocol is under review.
Setting them and the approval record is a distinct owner-approval step. No
held-out command may run while any value is ``None``. The finalized design
manifest itself remains byte-identical before and after approval. This file is
parsed as a restricted data record and is never imported or executed.
"""

APPROVED_MANIFEST_SHA256: str | None = "84d2c279cc00fc35f5d90e6f375c18e75950e4274519bcea16d3dbcda66af96a"
APPROVED_PROTOCOL_SHA256: str | None = "c8759d2d3cb347395d31c7f5b29f969eabf33973fbeb60ce865f5adf3455a79a"
APPROVED_SOURCE_LOCK_SHA256: str | None = "1bb2f24789944bd3ec61c2da522c42fde9b5f0a5142b234772baf0b6d54c5578"
OWNER_APPROVAL_DATE: str | None = "2026-09-10"
OWNER_APPROVAL_RECORD: str | None = "I approve these exact CTBF v5 confirmation hashes for held-out generation."
