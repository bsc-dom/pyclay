from queue import Queue

from dataclay import dclayMethod, DataClayObject

# TODO: I wanted to name this QueueAndListStream and then rename, but registration doesn't cope well with renaming
class StorageStream(DataClayObject):
    """StorageStream implementation by means of a queue and a list.

    This class implements the interface of StorageStream and internally
    it uses both a queue and a list in order to efficiently implement the
    blocking operations for the publish/poll behaviour and a list to allow
    a sensible get_all_elements response.

    @dclayImportFrom queue import Queue

    @ClassField alias str
    @ClassField closed bool
    @ClassField id anything
    @ClassField access_mode
    @ClassField object_list list
    """
    @dclayMethod(alias="str", internal_stream_info="anything")
    def __init__(self, alias=None, internal_stream_info=None):
        """Initialize and persist the stream."""

        # The following doesn't make a lot of sense, but follows closely the 
        # implementation proposed by @cramonco.
        self.alias = alias
        self.stream_type = "PERSISTENT"
        self.access_mode = "AT_MOST_ONCE"

        # Initialize the list, empty
        self.object_list = list()
        self.closed = False

        # Ensure that the object is persistent and perform EE initialization
        self.make_persistent(alias=alias)
        self._init_in_ee()

        self.id = self.getID()

    @dclayMethod(return_="str")
    def get_stream_id(self):
        """Returns the stream id.

        Defined by @cramonco in the original interface of the StorageStream.

        Its usefulness remains dubious IMHO (@abarcelo).
        """
        return self.id

    @dclayMethod(return_="str")
    def get_stream_alias(self):
        """Returns the stream alias.

        Defined by @cramonco in the original interface of the StorageStream.

        Its usefulness remains dubious IMHO (@abarcelo).
        """
        return self.alias

    @dclayMethod(return_="str")
    def get_stream_type(self):
        """Returns the internal stream type.

        Defined by @cramonco in the original interface of the StorageStream.

        Its usefulness remains dubious IMHO (@abarcelo).
        """
        return self.stream_type

    @dclayMethod()
    def _init_in_ee(self):
        """This should be in a method to ensure that it is executed in the EE."""
        # Set that it should be memory pinned
        self.set_memory_pinned(True)

        # And create the "volatile" queue (only in EE, not serialized)
        self.q = Queue()
        self.sentinel = object()

    @dclayMethod(obj="anything")    
    def publish(self, obj):
        """Publishes the given object on the stream.

        The object is put in both the queue (for future polls) and the list
        (for full iterations).

        Note that calling this method after close() is unsupported.
        """
        if self.closed:
            raise ValueError("The Stream is closed")
        self.object_list.append(obj)
        self.q.put(obj)

    @dclayMethod(return_="list")    
    def poll(self, num_objects=1):
        """Polls the given number of objects from the stream.
        
        This is a blocking operation.

        Note that calling this method after close() is unsupported.
        """
        if self.closed:
            raise ValueError("The Stream is closed")

        ret = list()
        for _ in range(num_objects):
            obj = self.q.get()

            if obj is self.sentinel:
                # The queue has been closed, put again the sentinel and raise
                self.q.put(obj)
                raise ValueError("The Stream is closed")

            ret.append(obj)
        return ret

    @dclayMethod()
    def close(self):
        """Closes the current stream."""
        # queue should not be used anymore, signal that it has been closed
        self.closed = True
        self.q.put(self.sentinel)

        # The stream can now be offloaded from memory if required by HeapManager.
        self.set_memory_pinned(False)

        # Note to the future troubleshooters:
        # If there is an ongoing "slow" thread doing __iter__, and meanwhile a close() is 
        # called and then the HeapManager decides to trigger and clean the stream before 
        # the iteration finishes, the self.q (inside __iter__ method) is undefined behaviour.

    @dclayMethod(return_="bool")
    def is_closed(self):
        """Returns whether the stream is closed or not."""
        return self.closed

    @dclayMethod(_local=True)
    def __iter__(self):
        """Iterate the enqueued elements (not the persistent list)."""
        value = self.q.get()

        while value is not self.sentinel:
            yield value
            value = self.q.get()

        # Ensure that the sentinel remains in place (maybe there is another poll blocked?)
        self.q.put(self.sentinel)

    @dclayMethod(return_="anything")
    def get_all_elems(self):
        """Return all elements in the stream.

        This method returns the list of elements available in the stream, including
        both objects already poll-ed and objects that have yet to be poll-ed.

        Note that this method works even after closing the stream.
        """
        return self.object_list
