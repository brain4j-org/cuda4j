package org.cuda4j;

import java.lang.foreign.*;
import java.lang.invoke.MethodHandle;

public interface CudaObject {
    
    long LONG_SIZE = Long.SIZE / 8; // 8
    long DOUBLE_SIZE = Double.SIZE / 8; // 8
    long INT_SIZE = Integer.SIZE / 8; // 4
    long FLOAT_SIZE = Float.SIZE / 8; // 4
    long SHORT_SIZE = Short.SIZE / 8; // 2
    long CHAR_SIZE = Character.SIZE / 8;
    long BYTE_SIZE = Byte.SIZE / 8; // 1
    Linker LINKER = Cuda4J.LINKER;
    SymbolLookup LOOKUP = Cuda4J.LOOKUP;
    MethodHandle CUDA_RELEASE_OBJECT = LINKER.downcallHandle(
        LOOKUP.findOrThrow("cuda_release_object"),
        FunctionDescriptor.ofVoid(ValueLayout.ADDRESS)
    );

    default void release() throws Throwable {
        CUDA_RELEASE_OBJECT.invokeExact(handle());
    }

    MemorySegment handle();
}