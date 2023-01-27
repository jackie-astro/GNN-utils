
SharedMemoryObject<SimulatorSyncronizedMessage> _sharedMemory;

/*******************************************/
struct SimulatorSyncronizedMessage : public SimulatorMessage
{
    void init()   
    {
        robotToSimSemaphore.init(0);  // 0   + post incre add
        simToRobotSemaphore.init(0);  // 0   - wait decre 
    }
    void waitForSimulator() { simToRobotSemaphore.decrement(); }
    void simulatorIsDone() { simToRobotSemaphore.increment(); }
    void waitForRobot() { robotToSimSemaphore.decrement(); }
    bool tryWaitForRobot() { return robotToSimSemaphore.tryDecrement(); }
    bool waitForRobotWithTimeout()
    {
        return robotToSimSemaphore.decrementTimeout(1, 0);
    }
    void robotIsDone() { robotToSimSemaphore.increment(); }

private:

    SharedMemorySemaphore robotToSimSemaphore, simToRobotSemaphore;
}

/*******************************************/
template <typename T>
class SharedMemoryObject
{
public:
    SharedMemoryObject() = default;   

    bool createNew(const std::string &name, bool allowOverwrite = false)
    {
        bool hadToDelete = false;
        assert(!_data);
        _name = name;
        _size = sizeof(T);
        printf("[Shared Memory] open new %s, size %ld bytes\n");

        // int shm_open(char* , )
        _fd = shm_open(name.c_str(), O_RDWR | O_CREAT,
                       S_IWUSR | S_IRUSR | S_IWGRP | S_IRGRP | S_IROTH);
        if (_fd == -1)
        {
            printf("[ERROR] SharedMemoryObject shm_open failed: %s\n",
                   strerror(errno));
            throw std::runtime_error("Failed to create shared memory!");
            return false;
        }

        struct stat s;

        if (fstat(_fd, &s))
        {
            printf("[ERROR] SharedMemoryObject::createNew(%s) stat: %s\n",
                   name.c_str(), strerror(errno));
            throw std::runtime_error("Failed to create shared memory!");
            return false;
        }

        if (s.st_size)
        {
            printf(
                "[Shared Memory] SharedMemoryObject::createNew(%s) on something that "
                "wasn't new (size is %ld bytes)\n",
                _name.c_str(), s.st_size);
            hadToDelete = true;
            if (!allowOverwrite)
                throw std::runtime_error(
                    "Failed to create shared memory - it already exists.");

            printf("\tusing existing shared memory!\n");
        }

        if (ftruncate(_fd, _size))
        {
            printf("[ERROR] SharedMemoryObject::createNew(%s) ftruncate(%ld): %s\n",
                   name.c_str(), _size, strerror(errno));
            throw std::runtime_error("Failed to create shared memory!");
            return false;
        }

        void *mem =
            mmap(nullptr, _size, PROT_READ | PROT_WRITE, MAP_SHARED, _fd, 0);


        if (mem == MAP_FAILED)
        {
            printf("[ERROR] SharedMemory::createNew(%s) mmap fail: %s\n",
                   _name.c_str(), strerror(errno));
            throw std::runtime_error("Failed to create shared memory!");
            return false;
        }

        memset(mem, 0, _size);

        _data = (T *)mem;
        return hadToDelete;
    }


    void attach(const std::string &name)
    {
        assert(!_data);
        _name = name;
        _size = sizeof(T);
        printf("[Shared Memory] open existing %s size %ld bytes\n", name.c_str(),
               _size);
        _fd = shm_open(name.c_str(), O_RDWR,
                       S_IWUSR | S_IRUSR | S_IWGRP | S_IRGRP | S_IROTH);
        if (_fd == -1)
        {
            printf("[ERROR] SharedMemoryObject::attach shm_open(%s) failed: %s\n",
                   _name.c_str(), strerror(errno));
            throw std::runtime_error("Failed to create shared memory!");
            return;
        }

        struct stat s;
        if (fstat(_fd, &s))
        {
            printf("[ERROR] SharedMemoryObject::attach(%s) stat: %s\n", name.c_str(),
                   strerror(errno));
            throw std::runtime_error("Failed to create shared memory!");
            return;
        }

        if ((size_t)s.st_size != _size)
        {
            printf(
                "[ERROR] SharedMemoryObject::attach(%s) on something that was "
                "incorrectly "
                "sized (size is %ld bytes, should be %ld)\n",
                _name.c_str(), s.st_size, _size);
            throw std::runtime_error("Failed to create shared memory!");
            return;
        }

        void *mem =
            mmap(nullptr, _size, PROT_READ | PROT_WRITE, MAP_SHARED, _fd, 0);
        if (mem == MAP_FAILED)
        {
            printf("[ERROR] SharedMemory::attach(%s) mmap fail: %s\n", _name.c_str(),
                   strerror(errno));
            throw std::runtime_error("Failed to create shared memory!");
            return;
        }

        _data = (T *)mem;
    }

    void closeNew()     // createNew
    {
        assert(_data);  // 断言


        // first, unmap
        if (munmap((void *)_data, _size))
        {
            printf("[ERROR] SharedMemoryObject::closeNew (%s) munmap %s\n",
                   _name.c_str(), strerror(errno));
            throw std::runtime_error("Failed to create shared memory!");
            return;
        }

        _data = nullptr;

        if (shm_unlink(_name.c_str()))
        {
            printf("[ERROR] SharedMemoryObject::closeNew (%s) shm_unlink %s\n",
                   _name.c_str(), strerror(errno));
            throw std::runtime_error("Failed to create shared memory!");
            return;
        }
        if (close(_fd))
        {
            printf("[ERROR] SharedMemoryObject::closeNew (%s) close %s\n",
                   _name.c_str(), strerror(errno));
            throw std::runtime_error("Failed to create shared memory!");
            return;
        }

        _fd = 0;
    }

    void detach()
    {
        assert(_data);

        if (munmap((void *)_data, _size))
        {
            printf("[ERROR] SharedMemoryObject::detach (%s) munmap %s\n",
                   _name.c_str(), strerror(errno));
            throw std::runtime_error("Failed to create shared memory!");
            return;
        }

        _data = nullptr;
        if (close(_fd))
        {
            printf("[ERROR] SharedMemoryObject::detach (%s) close %s\n",
                   _name.c_str(), strerror(errno));
            throw std::runtime_error("Failed to create shared memory!");
            return;
        }

        _fd = 0;
    }

    T *get()
    {
        assert(_data);
        return _data;
    }

    T &operator()()
    {
        assert(_data);
        return *_data;
    }

private:
    T* _data = nullptr;


    std::string _name;
    size_t _size;
    int _fd;
};
