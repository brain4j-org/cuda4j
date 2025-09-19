package org.cuda4j.device;

import org.cuda4j.CudaObject;
import org.cuda4j.context.CudaFunction;

import java.lang.foreign.Arena;
import java.lang.foreign.FunctionDescriptor;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.lang.invoke.MethodHandle;

public record CudaModule(MemorySegment handle) implements CudaObject {
    
    public static final MethodHandle CUDA_MODULE_LOAD = LINKER.downcallHandle(
        LOOKUP.findOrThrow("cuda_module_load"),
        FunctionDescriptor.of(ValueLayout.ADDRESS, ValueLayout.ADDRESS)
    );
    public static final MethodHandle CUDA_MODULE_LOAD_DATA = LINKER.downcallHandle(
        LOOKUP.findOrThrow("cuda_module_load_data"),
        FunctionDescriptor.of(ValueLayout.ADDRESS, ValueLayout.ADDRESS)
    );
    public static final MethodHandle CUDA_MODULE_UNLOAD = LINKER.downcallHandle(
        LOOKUP.findOrThrow("cuda_module_unload"),
        FunctionDescriptor.of(ValueLayout.JAVA_INT, ValueLayout.ADDRESS)
    );
    private static final MethodHandle CUDA_MODULE_GET_FUNCTION = LINKER.downcallHandle(
        LOOKUP.findOrThrow("cuda_module_get_function"),
        FunctionDescriptor.of(ValueLayout.ADDRESS, ValueLayout.ADDRESS, ValueLayout.ADDRESS)
    );
    
    public static CudaModule load(String path) throws Throwable {
        try (Arena arena = Arena.ofConfined()) {
            MemorySegment cPath = arena.allocateFrom(path);
            MemorySegment moduleHandle = (MemorySegment) CUDA_MODULE_LOAD.invoke(cPath);
            
            if (moduleHandle == null || moduleHandle.address() == 0) {
                throw new RuntimeException("cuModuleLoad failed for: " + path);
            }
            
            return new CudaModule(moduleHandle);
        }
    }
    
    public static CudaModule loadData(byte[] ptx) throws Throwable {
        try (Arena arena = Arena.ofConfined()) {
            MemorySegment data = arena.allocateFrom(ValueLayout.JAVA_BYTE, ptx);
            MemorySegment moduleHandle = (MemorySegment) CUDA_MODULE_LOAD_DATA.invoke(data);
            
            if (moduleHandle == null || moduleHandle.address() == 0) {
                throw new RuntimeException("cuModuleLoadData failed");
            }
            
            return new CudaModule(moduleHandle);
        }
    }
    
    public CudaFunction getFunction(String name) throws Throwable {
        try (Arena arena = Arena.ofConfined()) {
            MemorySegment cName = arena.allocateFrom(name);
            MemorySegment funcHandle = (MemorySegment) CUDA_MODULE_GET_FUNCTION.invoke(handle, cName);
            
            if (funcHandle == null || funcHandle.address() == 0) {
                throw new RuntimeException("Failed to get function: " + name);
            }
            
            return new CudaFunction(funcHandle);
        }
    }
    
    public void unload() throws Throwable {
        int result = (int) CUDA_MODULE_UNLOAD.invoke(handle);
        
        if (result != 0) {
            throw new RuntimeException("Failed to unload CUDA module (error code " + result + ")");
        }
    }
}
