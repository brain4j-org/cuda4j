package org.cuda4j.buffer;

import org.cuda4j.CudaObject;
import org.cuda4j.context.CudaStream;

import java.lang.foreign.*;
import java.lang.invoke.MethodHandle;

public record CudaBuffer(MemorySegment handle, long length) implements CudaObject {
    
    public static final MethodHandle CUDA_BUFFER_PTR = LINKER.downcallHandle(
        LOOKUP.findOrThrow("cuda_buffer_ptr"),
        FunctionDescriptor.of(ValueLayout.JAVA_LONG, ValueLayout.ADDRESS)
    );
    public static final MethodHandle CUDA_MEM_ALLOC = LINKER.downcallHandle(
        LOOKUP.findOrThrow("cuda_mem_alloc"),
        FunctionDescriptor.of(ValueLayout.ADDRESS, ValueLayout.JAVA_LONG)
    );
    public static final MethodHandle CUDA_MEM_FREE = LINKER.downcallHandle(
        LOOKUP.findOrThrow("cuda_mem_free"),
        FunctionDescriptor.of(ValueLayout.JAVA_INT, ValueLayout.ADDRESS)
    );
    public static final MethodHandle CUDA_MEMCPY_HTOD = LINKER.downcallHandle(
        LOOKUP.findOrThrow("cuda_memcpy_htod"),
        FunctionDescriptor.of(ValueLayout.JAVA_INT,
            ValueLayout.ADDRESS, // buffer pointer
            ValueLayout.ADDRESS, // host pointer
            ValueLayout.JAVA_LONG) // size
    );
    public static final MethodHandle CUDA_MEMCPY_DTOH = LINKER.downcallHandle(
        LOOKUP.findOrThrow("cuda_memcpy_dtoh"),
        FunctionDescriptor.of(ValueLayout.JAVA_INT,
            ValueLayout.ADDRESS, // host pointer
            ValueLayout.ADDRESS, // buffer pointer
            ValueLayout.JAVA_LONG) // size
    );
    public static final MethodHandle CUDA_MEMCPY_DTOD = LINKER.downcallHandle(
        LOOKUP.findOrThrow("cuda_memcpy_dtod"),
        FunctionDescriptor.of(ValueLayout.JAVA_INT,
            ValueLayout.ADDRESS, // destination pointer
            ValueLayout.ADDRESS, // source pointer
            ValueLayout.JAVA_LONG) // size
    );
    public static final MethodHandle CUDA_MEMCPY_HTOD_ASYNC = LINKER.downcallHandle(
        LOOKUP.findOrThrow("cuda_memcpy_htod_async"),
        FunctionDescriptor.of(ValueLayout.JAVA_INT,
            ValueLayout.ADDRESS, // buffer pointer
            ValueLayout.ADDRESS, // host pointer
            ValueLayout.JAVA_LONG, // size
            ValueLayout.ADDRESS) // stream
    );
    public static final MethodHandle CUDA_MEMCPY_DTOH_ASYNC = LINKER.downcallHandle(
        LOOKUP.findOrThrow("cuda_memcpy_dtoh_async"),
        FunctionDescriptor.of(ValueLayout.JAVA_INT,
            ValueLayout.ADDRESS, // host pointer
            ValueLayout.ADDRESS, // buffer pointer
            ValueLayout.JAVA_LONG, // size
            ValueLayout.ADDRESS) // stream
    );
    
    public static CudaBuffer allocate(float[] data, long size) throws Throwable {
        CudaBuffer buffer = allocate(size);
        buffer.copyToDevice(data);
        return buffer;
    }
    
    public static CudaBuffer allocate(float[] data, long size, CudaStream stream) throws Throwable {
        CudaBuffer buffer = allocate(size);
        buffer.copyToDeviceAsync(data, stream);
        return buffer;
    }
    
    public static CudaBuffer allocate(int[] data, long size) throws Throwable {
        CudaBuffer buffer = allocate(size);
        buffer.copyToDevice(data);
        return buffer;
    }
    
    public static CudaBuffer allocate(int[] data, long size, CudaStream stream) throws Throwable {
        CudaBuffer buffer = allocate(size);
        buffer.copyToDeviceAsync(data, stream);
        return buffer;
    }
    
    public static CudaBuffer allocate(long size) throws Throwable {
        MemorySegment ptr = (MemorySegment) CUDA_MEM_ALLOC.invoke(size);
        
        if (ptr == null || ptr.address() == 0) {
            throw new OutOfMemoryError("cuMemAlloc failed: " + ptr);
        }
        
        return new CudaBuffer(ptr, size);
    }
    
    public void transferTo(CudaBuffer destination, long size) throws Throwable {
        int res = (int) CUDA_MEMCPY_DTOH.invoke(destination.handle, handle, size);
        
        if (res != 0) {
            throw new RuntimeException("cuMemcpyDtoD failed: " + res);
        }
    }
    
    // copy to device
    public void copyToDevice(byte[] data) throws Throwable {
        try (Arena arena = Arena.ofConfined()) {
            MemorySegment host = arena.allocateFrom(ValueLayout.JAVA_BYTE, data);
            int res = (int) CUDA_MEMCPY_HTOD.invoke(handle, host, (long) data.length);
            
            if (res != 0) {
                throw new RuntimeException("cuMemcpyHtoD failed: " + res);
            }
        }
    }
    
    public void copyToDevice(int[] data) throws Throwable {
        try (Arena arena = Arena.ofConfined()) {
            MemorySegment host = arena.allocateFrom(ValueLayout.JAVA_INT, data);
            int res = (int) CUDA_MEMCPY_HTOD.invoke(handle, host, (long) data.length);
            
            if (res != 0) {
                throw new RuntimeException("cuMemcpyHtoD failed: " + res);
            }
        }
    }
    
    public void copyToDevice(float[] data) throws Throwable {
        try (Arena arena = Arena.ofConfined()) {
            MemorySegment host = arena.allocateFrom(ValueLayout.JAVA_FLOAT, data);
            int res = (int) CUDA_MEMCPY_HTOD.invoke(handle, host, (long) data.length);
            
            if (res != 0) {
                throw new RuntimeException("cuMemcpyHtoD failed: " + res);
            }
        }
    }
    
    // copy to host
    public void copyToHost(byte[] data) throws Throwable {
        try (Arena arena = Arena.ofConfined()) {
            MemorySegment host = arena.allocate(ValueLayout.JAVA_BYTE, (long) data.length);
            int res = (int) CUDA_MEMCPY_DTOH.invoke(host, handle, (long) data.length);
            
            if (res != 0) {
                throw new RuntimeException("cuMemcpyDtoH failed: " + res);
            }
            
            MemorySegment.copy(host, 0, MemorySegment.ofArray(data), 0, (long) data.length);
        }
    }
    
    public void copyToHost(int[] data) throws Throwable {
        try (Arena arena = Arena.ofConfined()) {
            MemorySegment host = arena.allocate(ValueLayout.JAVA_INT, (long) data.length);
            int res = (int) CUDA_MEMCPY_DTOH.invoke(host, handle, (long) data.length * Integer.BYTES);
            
            if (res != 0) {
                throw new RuntimeException("cuMemcpyDtoH failed: " + res);
            }
            
            MemorySegment.copy(host, 0, MemorySegment.ofArray(data), 0, (long) (long) data.length * Integer.BYTES);
        }
    }
    
    public void copyToHost(float[] data) throws Throwable {
        try (Arena arena = Arena.ofConfined()) {
            MemorySegment host = arena.allocate(ValueLayout.JAVA_FLOAT, (long) data.length);
            int res = (int) CUDA_MEMCPY_DTOH.invoke(host, handle, (long) data.length);
            
            if (res != 0) {
                throw new RuntimeException("cuMemcpyDtoH failed: " + res);
            }
            
            MemorySegment.copy(host, 0, MemorySegment.ofArray(data), 0, (long) data.length);
        }
    }
    
    // async copy to device
    public void copyToDeviceAsync(byte[] data, CudaStream stream) throws Throwable {
        try (Arena arena = Arena.ofConfined()) {
            MemorySegment host = arena.allocateFrom(ValueLayout.JAVA_BYTE, data);
            int res = (int) CUDA_MEMCPY_HTOD_ASYNC.invoke(handle, host, (long) data.length, stream.handle());
            
            if (res != 0) {
                throw new RuntimeException("cuMemcpyHtoDAsync failed: " + res);
            }
        }
    }
    
    public void copyToDeviceAsync(int[] data, CudaStream stream) throws Throwable {
        try (Arena arena = Arena.ofConfined()) {
            MemorySegment host = arena.allocateFrom(ValueLayout.JAVA_INT, data);
            int res = (int) CUDA_MEMCPY_HTOD_ASYNC.invoke(handle, host, (long) data.length, stream.handle());
            
            if (res != 0) {
                throw new RuntimeException("cuMemcpyHtoDAsync failed: " + res);
            }
        }
    }
    
    public void copyToDeviceAsync(float[] data, CudaStream stream) throws Throwable {
        try (Arena arena = Arena.ofConfined()) {
            MemorySegment host = arena.allocateFrom(ValueLayout.JAVA_FLOAT, data);
            int res = (int) CUDA_MEMCPY_HTOD_ASYNC.invoke(handle, host, (long) data.length, stream.handle());
            
            if (res != 0) {
                throw new RuntimeException("cuMemcpyHtoDAsync failed: " + res);
            }
        }
    }
    
    // async copy to host
    public void copyToHostAsync(byte[] data, CudaStream stream) throws Throwable {
        try (Arena arena = Arena.ofConfined()) {
            MemorySegment host = arena.allocate(ValueLayout.JAVA_BYTE, (long) data.length);
            int res = (int) CUDA_MEMCPY_DTOH_ASYNC.invoke(host, handle, (long) data.length, stream.handle());
            
            if (res != 0) {
                throw new RuntimeException("cuMemcpyDtoHAsync failed: " + res);
            }
            
            MemorySegment.copy(host, 0, MemorySegment.ofArray(data), 0, (long) data.length);
        }
    }
    
    public void copyToHostAsync(int[] data, CudaStream stream) throws Throwable {
        try (Arena arena = Arena.ofConfined()) {
            MemorySegment host = arena.allocate(ValueLayout.JAVA_INT, (long) data.length);
            int res = (int) CUDA_MEMCPY_DTOH_ASYNC.invoke(host, handle, (long) data.length * Integer.BYTES, stream.handle());
            if (res != 0) {
                throw new RuntimeException("cuMemcpyDtoHAsync failed: " + res);
            }
            MemorySegment.copy(host, 0, MemorySegment.ofArray(data), 0, (long) data.length * Integer.BYTES);
        }
    }
    
    public void copyToHostAsync(float[] data, CudaStream stream) throws Throwable {
        try (Arena arena = Arena.ofConfined()) {
            MemorySegment host = arena.allocate(ValueLayout.JAVA_FLOAT, (long) data.length);
            int res = (int) CUDA_MEMCPY_DTOH_ASYNC.invoke(host, handle, (long) data.length, stream.handle());
            
            if (res != 0) {
                throw new RuntimeException("cuMemcpyDtoHAsync failed: " + res);
            }
            
            MemorySegment.copy(host, 0, MemorySegment.ofArray(data), 0, (long) data.length);
        }
    }
    
    public long devicePointer() throws Throwable {
        return (long) CUDA_BUFFER_PTR.invoke(handle);
    }
}
