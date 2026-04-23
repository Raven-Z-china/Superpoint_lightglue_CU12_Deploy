/*
 * SPDX-FileCopyrightText: Copyright (c) 1993-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#ifndef TENSORRT_BUFFERS_H
#define TENSORRT_BUFFERS_H

#include "NvInfer.h"
#include "common.h"
#include "half.h"
#include <cassert>
#include <cuda_runtime_api.h>
#include <iostream>
#include <iterator>
#include <memory>
#include <new>
#include <numeric>
#include <string>
#include <vector>

namespace tensorrt_buffer
{

//!
//! \brief  The GenericBuffer class is a templated class for buffers.
//!
//! \details This templated RAII (Resource Acquisition Is Initialization) class handles the allocation,
//!          deallocation, querying of buffers on both the device and the host.
//!          It can handle data of arbitrary types because it stores byte buffers.
//!          The template parameters AllocFunc and FreeFunc are used for the
//!          allocation and deallocation of the buffer.
//!          AllocFunc must be a functor that takes in (void** ptr, size_t size)
//!          and returns bool. ptr is a pointer to where the allocated buffer address should be stored.
//!          size is the amount of memory in bytes to allocate.
//!          The boolean indicates whether or not the memory allocation was successful.
//!          FreeFunc must be a functor that takes in (void* ptr) and returns void.
//!          ptr is the allocated buffer address. It must work with nullptr input.
//!
    template <typename AllocFunc, typename FreeFunc>
    class GenericBuffer
    {
    public:
        //!
        //! \brief Construct an empty buffer.
        //!
        GenericBuffer(nvinfer1::DataType type = nvinfer1::DataType::kFLOAT)
                : mSize(0)
                , mCapacity(0)
                , mType(type)
                , mBuffer(nullptr)
        {
        }

        //!
        //! \brief Construct a buffer with the specified allocation size in bytes.
        //!
        GenericBuffer(size_t size, nvinfer1::DataType type)
                : mSize(size)
                , mCapacity(size)
                , mType(type)
        {
            if (!allocFn(&mBuffer, this->nbBytes()))
            {
                throw std::bad_alloc();
            }
        }

        GenericBuffer(GenericBuffer&& buf)
                : mSize(buf.mSize)
                , mCapacity(buf.mCapacity)
                , mType(buf.mType)
                , mBuffer(buf.mBuffer)
        {
            buf.mSize = 0;
            buf.mCapacity = 0;
            buf.mType = nvinfer1::DataType::kFLOAT;
            buf.mBuffer = nullptr;
        }

        GenericBuffer& operator=(GenericBuffer&& buf)
        {
            if (this != &buf)
            {
                freeFn(mBuffer);
                mSize = buf.mSize;
                mCapacity = buf.mCapacity;
                mType = buf.mType;
                mBuffer = buf.mBuffer;
                // Reset buf.
                buf.mSize = 0;
                buf.mCapacity = 0;
                buf.mBuffer = nullptr;
            }
            return *this;
        }

        //!
        //! \brief Returns pointer to underlying array.
        //!
        void* data()
        {
            return mBuffer;
        }

        //!
        //! \brief Returns pointer to underlying array.
        //!
        const void* data() const
        {
            return mBuffer;
        }

        //!
        //! \brief Returns the size (in number of elements) of the buffer.
        //!
        size_t size() const
        {
            return mSize;
        }

        //!
        //! \brief Returns the size (in bytes) of the buffer.
        //!
        size_t nbBytes() const
        {
            return this->size() * tensorrt_buffer::getElementSize(mType);
        }

        //!
        //! \brief Resizes the buffer. This is a no-op if the new size is smaller than or equal to the current capacity.
        //!
        void resize(size_t newSize)
        {
            mSize = newSize;
            if (mCapacity < newSize)
            {
                freeFn(mBuffer);
                if (!allocFn(&mBuffer, this->nbBytes()))
                {
                    throw std::bad_alloc{};
                }
                mCapacity = newSize;
            }
        }

        //!
        //! \brief Overload of resize that accepts Dims
        //!
        void resize(const nvinfer1::Dims& dims)
        {
            return this->resize(tensorrt_buffer::volume(dims));
        }

        ~GenericBuffer()
        {
            freeFn(mBuffer);
        }

    private:
        size_t mSize{0}, mCapacity{0};
        nvinfer1::DataType mType;
        void* mBuffer;
        AllocFunc allocFn;
        FreeFunc freeFn;
    };

    class DeviceAllocator
    {
    public:
        bool operator()(void** ptr, size_t size) const
        {
            return cudaMalloc(ptr, size) == cudaSuccess;
        }
    };

    class DeviceFree
    {
    public:
        void operator()(void* ptr) const
        {
            cudaFree(ptr);
        }
    };

    class HostAllocator
    {
    public:
        bool operator()(void** ptr, size_t size) const
        {
            *ptr = malloc(size);
            return *ptr != nullptr;
        }
    };

    class HostFree
    {
    public:
        void operator()(void* ptr) const
        {
            free(ptr);
        }
    };

    using DeviceBuffer = GenericBuffer<DeviceAllocator, DeviceFree>;
    using HostBuffer = GenericBuffer<HostAllocator, HostFree>;

//!
//! \brief  The ManagedBuffer class groups together a pair of corresponding device and host buffers.
//!
    class ManagedBuffer
    {
    public:
        DeviceBuffer deviceBuffer;
        HostBuffer hostBuffer;
    };

// TensorRT 10 compatibility: Helper functions
#if NV_TENSORRT_MAJOR >= 10
    inline int getBindingIndex(nvinfer1::ICudaEngine* engine, const char* tensorName)
    {
        int nbTensors = engine->getNbIOTensors();
        for (int i = 0; i < nbTensors; i++)
        {
            if (strcmp(engine->getIOTensorName(i), tensorName) == 0)
            {
                return i;
            }
        }
        return -1;
    }

    inline nvinfer1::Dims getBindingDimensions(nvinfer1::ICudaEngine* engine, int index)
    {
        return engine->getTensorShape(engine->getIOTensorName(index));
    }

    inline nvinfer1::Dims getBindingDimensions(nvinfer1::IExecutionContext* context, int index, nvinfer1::ICudaEngine* engine)
    {
        if (context)
        {
            return context->getTensorShape(engine->getIOTensorName(index));
        }
        return getBindingDimensions(engine, index);
    }

    inline nvinfer1::DataType getBindingDataType(nvinfer1::ICudaEngine* engine, int index)
    {
        return engine->getTensorDataType(engine->getIOTensorName(index));
    }

    inline int getBindingVectorizedDim(nvinfer1::ICudaEngine* engine, int index)
    {
        return engine->getTensorVectorizedDim(engine->getIOTensorName(index));
    }

    inline int getBindingComponentsPerElement(nvinfer1::ICudaEngine* engine, int index)
    {
        return engine->getTensorBytesPerComponent(engine->getIOTensorName(index)) / tensorrt_buffer::getElementSize(getBindingDataType(engine, index));
    }

    inline bool bindingIsInput(nvinfer1::ICudaEngine* engine, int index)
    {
        return engine->getTensorIOMode(engine->getIOTensorName(index)) == nvinfer1::TensorIOMode::kINPUT;
    }
#else
    // TensorRT 8/9 API - direct calls
    inline int getBindingIndex(nvinfer1::ICudaEngine* engine, const char* tensorName)
    {
        return engine->getBindingIndex(tensorName);
    }

    inline nvinfer1::Dims getBindingDimensions(nvinfer1::ICudaEngine* engine, int index)
    {
        return engine->getBindingDimensions(index);
    }

    inline nvinfer1::Dims getBindingDimensions(nvinfer1::IExecutionContext* context, int index, nvinfer1::ICudaEngine* engine)
    {
        return context ? context->getBindingDimensions(index) : engine->getBindingDimensions(index);
    }

    inline nvinfer1::DataType getBindingDataType(nvinfer1::ICudaEngine* engine, int index)
    {
        return engine->getBindingDataType(index);
    }

    inline int getBindingVectorizedDim(nvinfer1::ICudaEngine* engine, int index)
    {
        return engine->getBindingVectorizedDim(index);
    }

    inline int getBindingComponentsPerElement(nvinfer1::ICudaEngine* engine, int index)
    {
        return engine->getBindingComponentsPerElement(index);
    }

    inline bool bindingIsInput(nvinfer1::ICudaEngine* engine, int index)
    {
        return engine->bindingIsInput(index);
    }
#endif

//!
//! \brief  The BufferManager class handles host and device buffer allocation and deallocation.
//!
//! \details This RAII class handles host and device buffer allocation and deallocation,
//!          memcpy between host and device buffers to aid with inference,
//!          and debugging dumps to validate inference. The BufferManager class is meant to be
//!          used to simplify buffer management and any interactions between buffers and the engine.
//!
    class BufferManager
    {
    public:
        static const size_t kINVALID_SIZE_VALUE = ~size_t(0);

        //!
        //! \brief Create a BufferManager for handling buffer interactions with engine.
        //!
        BufferManager(std::shared_ptr<nvinfer1::ICudaEngine> engine, const int batchSize = 0,
                      const nvinfer1::IExecutionContext* context = nullptr)
                : mEngine(engine)
                , mBatchSize(batchSize)
                , mContext(context)
        {
            // Full Dims implies no batch size.
            assert(engine->hasImplicitBatchDimension() || mBatchSize == 0);
            // Create host and device buffers
#if NV_TENSORRT_MAJOR >= 10
            int nbBindings = mEngine->getNbIOTensors();
#else
            int nbBindings = mEngine->getNbBindings();
#endif
            for (int i = 0; i < nbBindings; i++)
            {
                // In TensorRT 10, use context dimensions if context is provided and shape has been set
#if NV_TENSORRT_MAJOR >= 10
                nvinfer1::Dims dims;
                bool isInput = bindingIsInput(mEngine.get(), i);
                if (mContext != nullptr) {
                    dims = mContext->getTensorShape(mEngine->getIOTensorName(i));
                } else {
                    dims = getBindingDimensions(mEngine.get(), i);
                }
                
                // For TensorRT 10 with dynamic shapes, if dims.nbDims is 0 or negative,
                // this is a dynamic output whose shape depends on input.
                // Allocate a reasonable maximum buffer size for these cases.
                if (dims.nbDims <= 0 && !isInput) {
                    // Use a large default buffer for dynamic outputs
                    // This assumes the output won't exceed ~2000x2000 matching matrix
                    dims.nbDims = 3;
                    dims.d[0] = 1;  // batch
                    dims.d[1] = 2000;  // max keypoints
                    dims.d[2] = 2000;  // max keypoints
                }
#else
                auto dims = getBindingDimensions(mEngine.get(), i);
#endif
                size_t vol = context || !mBatchSize ? 1 : static_cast<size_t>(mBatchSize);
                nvinfer1::DataType type = getBindingDataType(mEngine.get(), i);
                int vecDim = getBindingVectorizedDim(mEngine.get(), i);
                if (-1 != vecDim) // i.e., 0 != lgScalarsPerVector
                {
                    int scalarsPerVec = getBindingComponentsPerElement(mEngine.get(), i);
                    dims.d[vecDim] = divUp(dims.d[vecDim], scalarsPerVec);
                    vol *= scalarsPerVec;
                }
                vol *= tensorrt_buffer::volume(dims);
                std::unique_ptr<ManagedBuffer> manBuf{new ManagedBuffer()};
                manBuf->deviceBuffer = DeviceBuffer(vol, type);
                manBuf->hostBuffer = HostBuffer(vol, type);
                mDeviceBindings.emplace_back(manBuf->deviceBuffer.data());
                mManagedBuffers.emplace_back(std::move(manBuf));
            }
        }

        //!
        //! \brief Returns a vector of device buffers that you can use directly as
        //!        bindings for the execute and enqueue methods of IExecutionContext.
        //!
        std::vector<void*>& getDeviceBindings()
        {
            return mDeviceBindings;
        }

        //!
        //! \brief Returns a vector of device buffers.
        //!
        const std::vector<void*>& getDeviceBindings() const
        {
            return mDeviceBindings;
        }

        //!
        //! \brief Returns the device buffer corresponding to tensorName.
        //!        Returns nullptr if no such tensor can be found.
        //!
        void* getDeviceBuffer(const std::string& tensorName) const
        {
            return getBuffer(false, tensorName);
        }

        //!
        //! \brief Returns the host buffer corresponding to tensorName.
        //!        Returns nullptr if no such tensor can be found.
        //!
        void* getHostBuffer(const std::string& tensorName) const
        {
            return getBuffer(true, tensorName);
        }

        //!
        //! \brief Returns the size of the host and device buffers that correspond to tensorName.
        //!        Returns kINVALID_SIZE_VALUE if no such tensor can be found.
        //!
        size_t size(const std::string& tensorName) const
        {
            int index = getBindingIndex(mEngine.get(), tensorName.c_str());
            if (index == -1)
                return kINVALID_SIZE_VALUE;
            return mManagedBuffers[index]->hostBuffer.nbBytes();
        }

        //!
        //! \brief Templated print function that dumps buffers of arbitrary type to std::ostream.
        //!        rowCount parameter controls how many elements are on each line.
        //!        A rowCount of 1 means that there is only 1 element on each line.
        //!
        template <typename T>
        void print(std::ostream& os, void* buf, size_t bufSize, size_t rowCount)
        {
            assert(rowCount != 0);
            assert(bufSize % sizeof(T) == 0);
            T* typedBuf = static_cast<T*>(buf);
            size_t numItems = bufSize / sizeof(T);
            for (int i = 0; i < static_cast<int>(numItems); i++)
            {
                // Handle rowCount == 1 case
                if (rowCount == 1 && i != static_cast<int>(numItems) - 1)
                    os << typedBuf[i] << std::endl;
                else if (rowCount == 1)
                    os << typedBuf[i];
                    // Handle rowCount > 1 case
                else if (i % rowCount == 0)
                    os << typedBuf[i];
                else if (i % rowCount == rowCount - 1)
                    os << " " << typedBuf[i] << std::endl;
                else
                    os << " " << typedBuf[i];
            }
        }

        //!
        //! \brief Copy the contents of input host buffers to input device buffers synchronously.
        //!
        void copyInputToDevice()
        {
            memcpyBuffers(true, false, false);
        }

        //!
        //! \brief Copy the contents of output device buffers to output host buffers synchronously.
        //!
        void copyOutputToHost()
        {
            memcpyBuffers(false, true, false);
        }

        //!
        //! \brief Copy the contents of input host buffers to input device buffers asynchronously.
        //!
        void copyInputToDeviceAsync(const cudaStream_t& stream = 0)
        {
            memcpyBuffers(true, false, true, stream);
        }

        //!
        //! \brief Copy the contents of output device buffers to output host buffers asynchronously.
        //!
        void copyOutputToHostAsync(const cudaStream_t& stream = 0)
        {
            memcpyBuffers(false, true, true, stream);
        }

        ~BufferManager() = default;

    private:
        void* getBuffer(const bool isHost, const std::string& tensorName) const
        {
            int index = getBindingIndex(mEngine.get(), tensorName.c_str());
            if (index == -1)
                return nullptr;
            return (isHost ? mManagedBuffers[index]->hostBuffer.data() : mManagedBuffers[index]->deviceBuffer.data());
        }

        void memcpyBuffers(const bool copyInput, const bool deviceToHost, const bool async, const cudaStream_t& stream = 0)
        {
#if NV_TENSORRT_MAJOR >= 10
            int nbBindings = mEngine->getNbIOTensors();
#else
            int nbBindings = mEngine->getNbBindings();
#endif
            for (int i = 0; i < nbBindings; i++)
            {
                void* dstPtr
                        = deviceToHost ? mManagedBuffers[i]->hostBuffer.data() : mManagedBuffers[i]->deviceBuffer.data();
                const void* srcPtr
                        = deviceToHost ? mManagedBuffers[i]->deviceBuffer.data() : mManagedBuffers[i]->hostBuffer.data();
                const size_t byteSize = mManagedBuffers[i]->hostBuffer.nbBytes();
                const cudaMemcpyKind memcpyType = deviceToHost ? cudaMemcpyDeviceToHost : cudaMemcpyHostToDevice;
                if ((copyInput && bindingIsInput(mEngine.get(), i)) || (!copyInput && !bindingIsInput(mEngine.get(), i)))
                {
                    if (async)
                        CHECK(cudaMemcpyAsync(dstPtr, srcPtr, byteSize, memcpyType, stream));
                    else
                        CHECK(cudaMemcpy(dstPtr, srcPtr, byteSize, memcpyType));
                }
            }
        }

        std::shared_ptr<nvinfer1::ICudaEngine> mEngine;              //!< The pointer to the engine
        int mBatchSize;                                              //!< The batch size for legacy networks, 0 otherwise.
        const nvinfer1::IExecutionContext* mContext;                //!< The execution context for TensorRT 10 dynamic shapes
        std::vector<std::unique_ptr<ManagedBuffer>> mManagedBuffers; //!< The vector of pointers to managed buffers
        std::vector<void*> mDeviceBindings;                          //!< The vector of device buffers needed for engine execution
    };

} // namespace tensorrt_buffer

#endif // TENSORRT_BUFFERS_H
