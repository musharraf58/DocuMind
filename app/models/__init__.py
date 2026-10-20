from app.models.base import Base
from app.models.chunk import Chunk
from app.models.collection import Collection
from app.models.conversation import Conversation, Message, MessageRole, MessageStatus
from app.models.document import Document, DocumentStatus
from app.models.user import User

# Importing every model here registers it on Base.metadata, which is what
# Alembic autogenerate and the test suite's create_all() read.
__all__ = [
    "Base",
    "Chunk",
    "Collection",
    "Conversation",
    "Message",
    "MessageRole",
    "MessageStatus",
    "Document",
    "DocumentStatus",
    "User",
]
