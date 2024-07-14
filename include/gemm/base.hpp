#pragma once

#include <unordered_map>
#include <functional>
#include <string>
#include <variant>
#include <vector>
#include <stdexcept>
#include <driver_types.h>

namespace gemm {

    namespace base {

        class GemmOptions
        {
        public:
            using FieldType = std::variant<int, std::function<int()>>;
            

            void set(const std::string &key, int value)
            {
                if (fields_.find(key) == fields_.end()) {
                    throw std::runtime_error("Field " + key + " undefined");
                }
                fields_[key] = value;
            }

            void set(const std::string &key, std::function<int()> value)
            {
                if (fields_.find(key) == fields_.end()) {
                    throw std::runtime_error("Field " + key + " undefined");
                }
                fields_[key] = value;
            }

            int get(const std::string &key)
            {
                if (fields_.find(key) == fields_.end()) {
                    throw std::runtime_error("Field " + key + " undefined");
                }
                FieldType field = fields_[key];
                if (std::holds_alternative<int>(field))
                {
                    return std::get<int>(field);
                }
                else if (std::holds_alternative<std::function<int()>>(field))
                {
                    return std::get<std::function<int()>>(field)();
                }
            }

            std::unordered_map<std::string, FieldType> getFields() const
            {
                return fields_;
            }
        protected:
            void add(const std::string &key, int value)
            {
                keys_.push_back(key);
                fields_[key] = value;
            }

            void add(const std::string &key, std::function<int()> value)
            {
                keys_.push_back(key);
                fields_[key] = value;
            }
        private:
            std::unordered_map<std::string, FieldType> fields_;
            std::vector<std::string> keys_;
        };

        enum class Status {
            Success,
            Error
        };

        struct GemmOutput {
            Status status;
            cudaError_t code;
            std::string err = "";
            float excute_time_ms;
            GemmOutput() {}
            inline GemmOutput(cudaError_t code, float excute_time_ms)
            {
                this -> code = code;
                this -> excute_time_ms = excute_time_ms;
                this -> status = Status::Success;
                if (code != cudaSuccess) {
                    this -> status = Status::Error;
                    this -> err  = "CUDA error: " + std::string(cudaGetErrorString(code)) + "\n" + __FILE__ + ":" + std::to_string(__LINE__) + "\n";
                }
            }
        };
    }
}