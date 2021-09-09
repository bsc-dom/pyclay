from uuid import UUID

# ToDo: Once we deprecate 3.6, convert this to a **dataclass**
class ObjectInfo:
    object_id: UUID
    backend_id: UUID
    is_loaded: bool = None
    is_master: bool = None
    is_local: bool = None
