"""MPI runtime package.

Everything MPI-specific lives in this folder.

 - `Messaging_MPI.py`: MPI transport + message dispatch
 - `start_MPI.py`: MPI init + role startup
"""

from .Messaging_MPI import Message, MessageManager, MpiCommunicationManager, Observer
from .start_MPI import SplitNN_distributed, SplitNN_init

__all__ = [
    "Message",
    "MessageManager",
    "MpiCommunicationManager",
    "Observer",
    "SplitNN_init",
    "SplitNN_distributed",
]
