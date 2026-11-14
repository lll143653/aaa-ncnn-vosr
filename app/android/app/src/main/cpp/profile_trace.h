#pragma once
#include <android/trace.h>
#include <string>
#include <sstream>

struct TraceSpan {
    explicit TraceSpan(const std::string& name) { ATrace_beginSection(name.c_str()); }
    ~TraceSpan() { ATrace_endSection(); }
};
inline std::string jsonString(const std::string& text) {
    std::ostringstream out; out<<'"';
    const char* hex="0123456789abcdef";
    for (unsigned char c:text) {
        if (c=='"' || c=='\\') out<<'\\'<<char(c);
        else if (c<32) out<<"\\u00"<<hex[c>>4]<<hex[c&15];
        else out<<char(c);
    }
    out<<'"'; return out.str();
}
