#ifndef HLS_STREAM_H
#define HLS_STREAM_H

#include <queue>
#include <stdexcept>
#include <iostream>

namespace hls {
    template<typename T>
    class stream {
    private:
        std::queue<T> q;
    public:
        void write(const T& val) { q.push(val); }
        T read() { 
            if (q.empty()) {
                std::cerr << "Error: Reading from empty stream!" << std::endl;
                throw std::runtime_error("Empty stream");
            }
            T val = q.front(); 
            q.pop(); 
            return val; 
        }
        bool empty() const { return q.empty(); }
    };
}

#endif
