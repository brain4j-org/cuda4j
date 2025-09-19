package org.cuda4j;

import java.io.IOException;
import java.io.InputStream;
import java.lang.foreign.*;
import java.lang.invoke.MethodHandle;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardCopyOption;

public class Cuda4J {
    
    public static final Linker LINKER = Linker.nativeLinker();
    public static final SymbolLookup LOOKUP = loadFromResources("/libcuda4j.dll");
    public static final MethodHandle CUDA_INIT = LINKER.downcallHandle(
        LOOKUP.findOrThrow("cuda_init"),
        FunctionDescriptor.ofVoid()
    );
    public static final MethodHandle CUDA_DEVICE_COUNT = LINKER.downcallHandle(
        LOOKUP.findOrThrow("cuda_device_count"),
        FunctionDescriptor.of(ValueLayout.JAVA_INT)
    );

    public static SymbolLookup loadFromResources(String resourceName) {
        try (InputStream in = Cuda4J.class.getResourceAsStream(resourceName)) {
            if (in == null) {
                throw new IllegalArgumentException("Resource not found: " + resourceName);
            }
            
            String suffix = resourceName.substring(resourceName.lastIndexOf('.'));
            Path tempFile = Files.createTempFile("nativeLib", suffix);
            
            Files.copy(in, tempFile, StandardCopyOption.REPLACE_EXISTING);
            tempFile.toFile().deleteOnExit();
            
            return SymbolLookup.libraryLookup(tempFile.toString(), Arena.global());
        } catch (IOException e) {
            throw new RuntimeException(e);
        }
    }
    
    public static void init() throws Throwable {
        CUDA_INIT.invokeExact();
    }
    
    public static int getDeviceCount() throws Throwable {
        return (int) CUDA_DEVICE_COUNT.invokeExact();
    }
}
