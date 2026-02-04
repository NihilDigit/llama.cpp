-keep class com.nihildigit.lightwayllama.* { *; }
-keep class com.nihildigit.lightwayllama.gguf.* { *; }

-keepclasseswithmembernames class * {
    native <methods>;
}

-keep class kotlin.Metadata { *; }
