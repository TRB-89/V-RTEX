#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <map>
#include <variant>
#include <chrono>
#include <cmath>
#include <iomanip>
#include <stdexcept>
#include <cctype>

#include "vortex/matrix.hpp"
#include "vortex/sparse_matrix.hpp"
#include "vortex/revised_simplex.hpp"
#include "vortex/sparse_revised_simplex.hpp"
#include "vortex/milp.hpp"
#include "vortex/qp.hpp"
#include "vortex/cuda_backend.hpp"

namespace {

// ==============================================================================
// Lightweight, Robust JSON Value Representation & Parser
// ==============================================================================
struct JsonValue;
using JsonObject = std::map<std::string, JsonValue>;
using JsonArray = std::vector<JsonValue>;

struct JsonValue {
    enum class Type { Null, Bool, Number, String, Array, Object } type{Type::Null};
    bool bool_val{false};
    double num_val{0.0};
    std::string str_val;
    JsonArray arr_val;
    JsonObject obj_val;

    bool is_null() const { return type == Type::Null; }
    bool is_bool() const { return type == Type::Bool; }
    bool is_number() const { return type == Type::Number; }
    bool is_string() const { return type == Type::String; }
    bool is_array() const { return type == Type::Array; }
    bool is_object() const { return type == Type::Object; }

    const JsonValue& operator[](const std::string& key) const {
        static const JsonValue null_val;
        if (type != Type::Object) return null_val;
        auto it = obj_val.find(key);
        if (it != obj_val.end()) return it->second;
        return null_val;
    }

    const JsonValue& operator[](std::size_t idx) const {
        static const JsonValue null_val;
        if (type != Type::Array || idx >= arr_val.size()) return null_val;
        return arr_val[idx];
    }

    double as_double(double default_val = 0.0) const {
        return (type == Type::Number) ? num_val : default_val;
    }

    int as_int(int default_val = 0) const {
        return (type == Type::Number) ? static_cast<int>(num_val) : default_val;
    }

    bool as_bool(bool default_val = false) const {
        return (type == Type::Bool) ? bool_val : default_val;
    }

    std::string as_string(const std::string& default_val = "") const {
        return (type == Type::String) ? str_val : default_val;
    }
};

class JsonParser {
public:
    explicit JsonParser(const std::string& src) : src_(src), pos_(0) {}

    JsonValue parse() {
        skip_whitespace();
        if (pos_ >= src_.size()) return JsonValue{};
        return parse_value();
    }

private:
    const std::string& src_;
    std::size_t pos_;

    void skip_whitespace() {
        while (pos_ < src_.size()) {
            char c = src_[pos_];
            if (c == ' ' || c == '\t' || c == '\n' || c == '\r') {
                pos_++;
            } else if (c == '/' && pos_ + 1 < src_.size() && src_[pos_ + 1] == '/') {
                pos_ += 2;
                while (pos_ < src_.size() && src_[pos_] != '\n') pos_++;
            } else {
                break;
            }
        }
    }

    JsonValue parse_value() {
        skip_whitespace();
        if (pos_ >= src_.size()) return JsonValue{};
        char c = src_[pos_];

        if (c == '{') return parse_object();
        if (c == '[') return parse_array();
        if (c == '"') return parse_string();
        if (c == 't' || c == 'f') return parse_bool();
        if (c == 'n') return parse_null();
        if (c == '-' || std::isdigit(static_cast<unsigned char>(c))) return parse_number();

        throw std::runtime_error(std::string("Unexpected character in JSON: '") + c + "' at position " + std::to_string(pos_));
    }

    JsonValue parse_object() {
        JsonValue val;
        val.type = JsonValue::Type::Object;
        pos_++; // skip '{'
        skip_whitespace();

        if (pos_ < src_.size() && src_[pos_] == '}') {
            pos_++;
            return val;
        }

        while (pos_ < src_.size()) {
            skip_whitespace();
            if (pos_ >= src_.size() || src_[pos_] != '"') {
                throw std::runtime_error("Expected string key in JSON object at pos " + std::to_string(pos_));
            }
            std::string key = parse_string().str_val;
            skip_whitespace();
            if (pos_ >= src_.size() || src_[pos_] != ':') {
                throw std::runtime_error("Expected ':' after key in JSON object at pos " + std::to_string(pos_));
            }
            pos_++; // skip ':'
            val.obj_val[key] = parse_value();
            skip_whitespace();
            if (pos_ < src_.size() && src_[pos_] == ',') {
                pos_++;
                continue;
            }
            if (pos_ < src_.size() && src_[pos_] == '}') {
                pos_++;
                break;
            }
            throw std::runtime_error("Expected ',' or '}' in JSON object at pos " + std::to_string(pos_));
        }
        return val;
    }

    JsonValue parse_array() {
        JsonValue val;
        val.type = JsonValue::Type::Array;
        pos_++; // skip '['
        skip_whitespace();

        if (pos_ < src_.size() && src_[pos_] == ']') {
            pos_++;
            return val;
        }

        while (pos_ < src_.size()) {
            val.arr_val.push_back(parse_value());
            skip_whitespace();
            if (pos_ < src_.size() && src_[pos_] == ',') {
                pos_++;
                continue;
            }
            if (pos_ < src_.size() && src_[pos_] == ']') {
                pos_++;
                break;
            }
            throw std::runtime_error("Expected ',' or ']' in JSON array at pos " + std::to_string(pos_));
        }
        return val;
    }

    JsonValue parse_string() {
        JsonValue val;
        val.type = JsonValue::Type::String;
        pos_++; // skip opening '"'
        std::string s;
        while (pos_ < src_.size()) {
            char c = src_[pos_++];
            if (c == '"') {
                val.str_val = s;
                return val;
            }
            if (c == '\\' && pos_ < src_.size()) {
                char esc = src_[pos_++];
                switch (esc) {
                    case '"': s += '"'; break;
                    case '\\': s += '\\'; break;
                    case '/': s += '/'; break;
                    case 'b': s += '\b'; break;
                    case 'f': s += '\f'; break;
                    case 'n': s += '\n'; break;
                    case 'r': s += '\r'; break;
                    case 't': s += '\t'; break;
                    default: s += esc; break;
                }
            } else {
                s += c;
            }
        }
        throw std::runtime_error("Unterminated string in JSON");
    }

    JsonValue parse_number() {
        JsonValue val;
        val.type = JsonValue::Type::Number;
        std::size_t start = pos_;
        if (src_[pos_] == '-') pos_++;
        while (pos_ < src_.size() && std::isdigit(static_cast<unsigned char>(src_[pos_]))) pos_++;
        if (pos_ < src_.size() && src_[pos_] == '.') {
            pos_++;
            while (pos_ < src_.size() && std::isdigit(static_cast<unsigned char>(src_[pos_]))) pos_++;
        }
        if (pos_ < src_.size() && (src_[pos_] == 'e' || src_[pos_] == 'E')) {
            pos_++;
            if (pos_ < src_.size() && (src_[pos_] == '+' || src_[pos_] == '-')) pos_++;
            while (pos_ < src_.size() && std::isdigit(static_cast<unsigned char>(src_[pos_]))) pos_++;
        }
        std::string num_str = src_.substr(start, pos_ - start);
        val.num_val = std::stod(num_str);
        return val;
    }

    JsonValue parse_bool() {
        JsonValue val;
        val.type = JsonValue::Type::Bool;
        if (src_.compare(pos_, 4, "true") == 0) {
            val.bool_val = true;
            pos_ += 4;
            return val;
        }
        if (src_.compare(pos_, 5, "false") == 0) {
            val.bool_val = false;
            pos_ += 5;
            return val;
        }
        throw std::runtime_error("Invalid boolean at pos " + std::to_string(pos_));
    }

    JsonValue parse_null() {
        if (src_.compare(pos_, 4, "null") == 0) {
            pos_ += 4;
            JsonValue val;
            val.type = JsonValue::Type::Null;
            return val;
        }
        throw std::runtime_error("Invalid literal at pos " + std::to_string(pos_));
    }
};

// ==============================================================================
// JSON Serialization Helpers
// ==============================================================================
std::string escape_json_string(const std::string& s) {
    std::string out = "\"";
    for (char c : s) {
        switch (c) {
            case '"': out += "\\\""; break;
            case '\\': out += "\\\\"; break;
            case '\b': out += "\\b"; break;
            case '\f': out += "\\f"; break;
            case '\n': out += "\\n"; break;
            case '\r': out += "\\r"; break;
            case '\t': out += "\\t"; break;
            default:
                if (static_cast<unsigned char>(c) < 32) {
                    char buf[8];
                    std::snprintf(buf, sizeof(buf), "\\u%04x", static_cast<unsigned char>(c));
                    out += buf;
                } else {
                    out += c;
                }
                break;
        }
    }
    out += "\"";
    return out;
}

std::string format_double(double d) {
    if (std::isnan(d)) return "\"NaN\"";
    if (std::isinf(d)) return (d > 0) ? "\"Infinity\"" : "\"-Infinity\"";
    std::ostringstream ss;
    ss << std::setprecision(12) << d;
    return ss.str();
}

std::string format_vector(const std::vector<double>& v) {
    std::string s = "[";
    for (std::size_t i = 0; i < v.size(); ++i) {
        if (i > 0) s += ", ";
        s += format_double(v[i]);
    }
    s += "]";
    return s;
}

std::string format_size_t_vector(const std::vector<std::size_t>& v) {
    std::string s = "[";
    for (std::size_t i = 0; i < v.size(); ++i) {
        if (i > 0) s += ", ";
        s += std::to_string(v[i]);
    }
    s += "]";
    return s;
}

std::string format_device_diagnostics(const vortex::SolverDeviceDiagnostics& d) {
    std::ostringstream ss;
    ss << "{\n";
    ss << "    \"requested_backend\": " << escape_json_string(vortex::to_string(d.requested_backend)) << ",\n";
    ss << "    \"executed_backend\": " << escape_json_string(vortex::to_string(d.executed_backend)) << ",\n";
    ss << "    \"gpu_iterations\": " << d.gpu_iterations << ",\n";
    ss << "    \"gpu_gemv_calls\": " << d.gpu_gemv_calls << ",\n";
    ss << "    \"gpu_spmv_calls\": " << d.gpu_spmv_calls << ",\n";
    ss << "    \"h2d_transfers\": " << d.h2d_transfers << ",\n";
    ss << "    \"d2h_transfers\": " << d.d2h_transfers << ",\n";
    ss << "    \"fallback_reason\": " << escape_json_string(d.fallback_reason) << "\n";
    ss << "  }";
    return ss.str();
}

std::string to_string(vortex::MILPStatus status) {
    switch (status) {
        case vortex::MILPStatus::OPTIMAL: return "OPTIMAL";
        case vortex::MILPStatus::FEASIBLE_LIMIT: return "FEASIBLE_LIMIT";
        case vortex::MILPStatus::INFEASIBLE: return "INFEASIBLE";
        case vortex::MILPStatus::UNBOUNDED: return "UNBOUNDED";
        case vortex::MILPStatus::NODE_LIMIT: return "NODE_LIMIT";
        case vortex::MILPStatus::TIME_LIMIT: return "TIME_LIMIT";
        case vortex::MILPStatus::NUMERICAL_ERROR: return "NUMERICAL_ERROR";
        default: return "UNKNOWN";
    }
}

// ==============================================================================
// Hardware and Device Telemetry JSON
// ==============================================================================
std::string get_device_info_json() {
    std::ostringstream ss;
    ss << "{\n";
    ss << "  \"cuda_available\": " << (vortex::cuda::is_cuda_available() ? "true" : "false") << ",\n";
    if (vortex::cuda::is_cuda_available()) {
        auto dev = vortex::cuda::get_cuda_device_info(0);
        ss << "  \"gpu\": {\n";
        ss << "    \"device_id\": " << dev.device_id << ",\n";
        ss << "    \"name\": " << escape_json_string(dev.name) << ",\n";
        ss << "    \"compute_capability\": \"" << dev.major << "." << dev.minor << "\",\n";
        ss << "    \"multiprocessor_count\": " << dev.multiprocessor_count << ",\n";
        ss << "    \"total_memory_bytes\": " << dev.total_memory_bytes << ",\n";
        ss << "    \"free_memory_bytes\": " << dev.free_memory_bytes << ",\n";
        ss << "    \"max_threads_per_block\": " << dev.max_threads_per_block << ",\n";
        ss << "    \"warp_size\": " << dev.warp_size << "\n";
        ss << "  },\n";
    } else {
        ss << "  \"gpu\": null,\n";
    }
    ss << "  \"supported_solvers\": [\"LP_DENSE\", \"LP_SPARSE\", \"MILP\", \"QP\"],\n";
    ss << "  \"supported_backends\": [\"CPU\", \"GPU\", \"AUTO\"],\n";
    ss << "  \"system_info\": {\n";
    ss << "    \"solver_name\": \"VØRTEX Optimization Engine\",\n";
    ss << "    \"version\": \"3.0.0\",\n";
    ss << "    \"cxx_standard\": \"C++20\",\n";
    ss << "    \"precision\": \"IEEE 754 Double Precision (FP64)\"\n";
    ss << "  }\n";
    ss << "}\n";
    return ss.str();
}

std::string get_capabilities_json() {
    std::ostringstream ss;
    ss << "{\n";
    ss << "  \"engine\": \"VØRTEX Optimization Engine\",\n";
    ss << "  \"version\": \"3.0.0\",\n";
    ss << "  \"cxx_standard\": \"C++20\",\n";
    ss << "  \"precision\": \"IEEE 754 FP64 (Double Precision)\",\n";
    ss << "  \"solvers\": {\n";
    ss << "    \"LP\": {\n";
    ss << "      \"algorithms\": [\"Dense Revised Simplex (LU Factorization)\", \"Sparse Revised Simplex (CSR/CSC)\", \"Two-Phase Standard Simplex\"],\n";
    ss << "      \"senses\": [\"MAXIMIZE\", \"MINIMIZE\"],\n";
    ss << "      \"constraints\": [\"LEQ (<=)\", \"GEQ (>=)\", \"EQ (==)\"],\n";
    ss << "      \"backends\": [\"CPU\", \"GPU\", \"AUTO\"],\n";
    ss << "      \"status_codes\": [\"OPTIMAL\", \"INFEASIBLE\", \"UNBOUNDED\", \"ITERATION_LIMIT\", \"NUMERICAL_ERROR\", \"INVALID_INPUT\"]\n";
    ss << "    },\n";
    ss << "    \"MILP\": {\n";
    ss << "      \"algorithms\": [\"Branch-and-Bound with Linear Relaxations\"],\n";
    ss << "      \"variable_types\": [\"CONTINUOUS\", \"INTEGER\", \"BINARY\"],\n";
    ss << "      \"branching_strategies\": [\"MOST_FRACTIONAL\", \"FIRST_FRACTIONAL\"],\n";
    ss << "      \"node_selection\": [\"BEST_BOUND\", \"DEPTH_FIRST\", \"BREADTH_FIRST\"],\n";
    ss << "      \"backends\": [\"CPU\"],\n";
    ss << "      \"status_codes\": [\"OPTIMAL\", \"FEASIBLE_LIMIT\", \"INFEASIBLE\", \"UNBOUNDED\", \"NODE_LIMIT\", \"TIME_LIMIT\", \"NUMERICAL_ERROR\"]\n";
    ss << "    },\n";
    ss << "    \"QP\": {\n";
    ss << "      \"algorithms\": [\"Primal-Dual Predictor-Corrector Interior-Point Method with Dynamic Tikhonov Regularization\"],\n";
    ss << "      \"problem_classes\": [\"Convex strictly positive definite Q\", \"Convex positive semidefinite singular Q\", \"Linear objective Q=0\", \"Unconstrained QP (m=0)\"],\n";
    ss << "      \"constraints\": [\"Linear Inequalities A*x <= b\"],\n";
    ss << "      \"backends\": [\"CPU\", \"GPU\", \"AUTO\"],\n";
    ss << "      \"status_codes\": [\"OPTIMAL\", \"MAX_ITERATIONS\", \"NUMERICAL_ERROR\", \"UNBOUNDED\", \"INFEASIBLE\", \"INVALID_INPUT\", \"UNSUPPORTED\"]\n";
    ss << "    }\n";
    ss << "  },\n";
    ss << "  \"supported_file_formats\": {\n";
    ss << "    \"JSON\": true,\n";
    ss << "    \"MPS\": false,\n";
    ss << "    \"LP_FILE\": false\n";
    ss << "  },\n";
    ss << "  \"cuda\": {\n";
    ss << "    \"available\": " << (vortex::cuda::is_cuda_available() ? "true" : "false") << ",\n";
    if (vortex::cuda::is_cuda_available()) {
        auto dev = vortex::cuda::get_cuda_device_info(0);
        ss << "    \"device_name\": " << escape_json_string(dev.name) << ",\n";
        ss << "    \"compute_capability\": \"" << dev.major << "." << dev.minor << "\",\n";
        ss << "    \"multiprocessors\": " << dev.multiprocessor_count << ",\n";
        ss << "    \"total_vram_bytes\": " << dev.total_memory_bytes << ",\n";
        ss << "    \"kernels\": [\"cuBLAS GEMV (FP64)\", \"cuSPARSE SpMV (CSR FP64)\"]\n";
    } else {
        ss << "    \"device_name\": null,\n";
        ss << "    \"kernels\": []\n";
    }
    ss << "  },\n";
    ss << "  \"limits\": {\n";
    ss << "    \"max_variables\": 50000,\n";
    ss << "    \"max_constraints\": 50000,\n";
    ss << "    \"max_elements\": 25000000\n";
    ss << "  }\n";
    ss << "}\n";
    return ss.str();
}

} // anonymous namespace

int main(int argc, char* argv[]) {
    try {
        std::string input_path;
        std::string output_path;
        bool device_info_only = false;
        bool capabilities_only = false;

        for (int i = 1; i < argc; ++i) {
            std::string arg = argv[i];
            if (arg == "--device-info") {
                device_info_only = true;
            } else if (arg == "--capabilities") {
                capabilities_only = true;
            } else if (arg == "--input" && i + 1 < argc) {
                input_path = argv[++i];
            } else if (arg == "--output" && i + 1 < argc) {
                output_path = argv[++i];
            } else if (arg == "--stdin") {
                input_path = "-";
            } else if (arg == "--stdout") {
                output_path = "-";
            } else if (arg == "--help" || arg == "-h") {
                std::cout << "VØRTEX Native Solver Runner\n";
                std::cout << "Usage:\n";
                std::cout << "  vortex_solver_runner --capabilities\n";
                std::cout << "  vortex_solver_runner --device-info\n";
                std::cout << "  vortex_solver_runner --input <problem.json> --output <result.json>\n";
                std::cout << "  vortex_solver_runner --stdin --stdout\n";
                return 0;
            }
        }

        if (capabilities_only) {
            std::string caps = get_capabilities_json();
            if (!output_path.empty() && output_path != "-") {
                std::ofstream out(output_path);
                if (!out) {
                    std::cerr << "Failed to open output file: " << output_path << "\n";
                    return 1;
                }
                out << caps;
            } else {
                std::cout << caps;
            }
            return 0;
        }

        if (device_info_only) {
            std::string info = get_device_info_json();
            if (!output_path.empty() && output_path != "-") {
                std::ofstream out(output_path);
                if (!out) {
                    std::cerr << "Failed to open output file: " << output_path << "\n";
                    return 1;
                }
                out << info;
            } else {
                std::cout << info;
            }
            return 0;
        }

        if (input_path.empty() || output_path.empty()) {
            std::cerr << "Error: Both --input and --output arguments (or --stdin / --stdout) are required.\n";
            return 1;
        }

        // Read input JSON
        std::string json_str;
        if (input_path == "-" || input_path == "stdin") {
            std::string line;
            while (std::getline(std::cin, line)) {
                json_str += line + "\n";
            }
        } else {
            std::ifstream in(input_path);
            if (!in) {
                std::cerr << "Failed to open input file: " << input_path << "\n";
                return 1;
            }
            std::stringstream buffer;
            buffer << in.rdbuf();
            json_str = buffer.str();
        }

        JsonParser parser(json_str);
        JsonValue root = parser.parse();

        std::string solver_type = root["solver_type"].as_string("LP");
        std::string backend_str = root["backend"].as_string("AUTO");
        vortex::SolverBackend requested_backend = vortex::SolverBackend::AUTO;
        if (backend_str == "CPU") requested_backend = vortex::SolverBackend::CPU;
        else if (backend_str == "GPU") requested_backend = vortex::SolverBackend::GPU;

        auto t_start = std::chrono::high_resolution_clock::now();
        std::ostringstream out_json;
        out_json << "{\n";

        if (solver_type == "LP") {
            // Linear Program
            std::string sense_str = root["sense"].as_string("MAXIMIZE");
            vortex::ObjectiveSense sense = (sense_str == "MINIMIZE") 
                ? vortex::ObjectiveSense::MINIMIZE 
                : vortex::ObjectiveSense::MAXIMIZE;

            // Read objective c
            std::vector<double> c;
            for (const auto& item : root["c"].arr_val) {
                c.push_back(item.as_double(0.0));
            }

            // Read constraint matrix A
            std::size_t rows = root["A"].arr_val.size();
            std::size_t cols = (rows > 0) ? root["A"].arr_val[0].arr_val.size() : 0;
            vortex::Matrix<double> A(rows, cols);
            std::size_t nnz = 0;
            for (std::size_t i = 0; i < rows; ++i) {
                for (std::size_t j = 0; j < cols; ++j) {
                    double v = root["A"].arr_val[i].arr_val[j].as_double(0.0);
                    A(i, j) = v;
                    if (std::abs(v) > 1e-15) nnz++;
                }
            }

            // Read b
            std::vector<double> b;
            for (const auto& item : root["b"].arr_val) {
                b.push_back(item.as_double(0.0));
            }

            // Constraint types
            std::vector<vortex::ConstraintType> constraint_types;
            if (root["constraint_types"].is_array() && root["constraint_types"].arr_val.size() == rows) {
                for (const auto& ct : root["constraint_types"].arr_val) {
                    std::string s = ct.as_string("LEQ");
                    if (s == "GEQ") constraint_types.push_back(vortex::ConstraintType::GEQ);
                    else if (s == "EQ") constraint_types.push_back(vortex::ConstraintType::EQ);
                    else constraint_types.push_back(vortex::ConstraintType::LEQ);
                }
            } else {
                constraint_types.assign(rows, vortex::ConstraintType::LEQ);
            }

            // Options
            vortex::RevisedSimplexOptions opts;
            opts.backend = requested_backend;
            if (root["options"].is_object()) {
                const auto& o = root["options"];
                if (o["tolerance"].is_number()) opts.tolerance = o["tolerance"].as_double();
                if (o["max_iterations"].is_number()) opts.max_iterations = static_cast<std::size_t>(o["max_iterations"].as_int());
                if (o["pivot_rule"].is_string()) {
                    std::string pr = o["pivot_rule"].as_string();
                    if (pr == "BLAND") opts.pivot_rule = vortex::PivotRule::BLAND;
                    else opts.pivot_rule = vortex::PivotRule::DANTZIG;
                }
            }

            vortex::RevisedSimplexSolver solver(opts);
            vortex::RevisedSimplexResult result = solver.solve(c, A, b, constraint_types, sense);

            auto t_end = std::chrono::high_resolution_clock::now();
            double wall_time_ms = std::chrono::duration<double, std::milli>(t_end - t_start).count();

            out_json << "  \"success\": true,\n";
            out_json << "  \"solver_type\": \"LP\",\n";
            out_json << "  \"status\": " << escape_json_string(vortex::to_string(result.status)) << ",\n";
            out_json << "  \"message\": " << escape_json_string(result.message) << ",\n";
            out_json << "  \"objective_value\": " << format_double(result.objective_value) << ",\n";
            out_json << "  \"variable_values\": " << format_vector(result.variable_values) << ",\n";
            out_json << "  \"slack_values\": " << format_vector(result.slack_values) << ",\n";
            out_json << "  \"surplus_values\": " << format_vector(result.surplus_values) << ",\n";
            out_json << "  \"dual_values\": " << format_vector(result.dual_values) << ",\n";
            out_json << "  \"reduced_costs\": " << format_vector(result.reduced_costs) << ",\n";
            out_json << "  \"basis_indices\": " << format_size_t_vector(result.basis_indices) << ",\n";
            out_json << "  \"iterations\": " << result.iterations << ",\n";
            out_json << "  \"phase1_iterations\": " << result.phase1_iterations << ",\n";
            out_json << "  \"phase2_iterations\": " << result.phase2_iterations << ",\n";
            out_json << "  \"max_primal_residual\": " << format_double(result.max_primal_residual) << ",\n";
            out_json << "  \"max_dual_residual\": " << format_double(result.max_dual_residual) << ",\n";
            out_json << "  \"basis_condition\": " << format_double(result.basis_condition) << ",\n";
            out_json << "  \"wall_time_ms\": " << format_double(wall_time_ms) << ",\n";
            out_json << "  \"device_diagnostics\": " << format_device_diagnostics(result.device_diagnostics) << ",\n";
            out_json << "  \"matrix_info\": {\n";
            out_json << "    \"rows\": " << rows << ",\n";
            out_json << "    \"cols\": " << cols << ",\n";
            out_json << "    \"nnz\": " << nnz << ",\n";
            out_json << "    \"density\": " << format_double((rows * cols > 0) ? (static_cast<double>(nnz) / (rows * cols)) : 0.0) << "\n";
            out_json << "  }\n";
        }
        else if (solver_type == "MILP") {
            // Mixed Integer Linear Program
            std::string sense_str = root["sense"].as_string("MAXIMIZE");
            vortex::ObjectiveSense sense = (sense_str == "MINIMIZE") 
                ? vortex::ObjectiveSense::MINIMIZE 
                : vortex::ObjectiveSense::MAXIMIZE;

            vortex::MixedIntegerLinearProgram milp;
            milp.sense = sense;

            for (const auto& item : root["c"].arr_val) {
                milp.c.push_back(item.as_double(0.0));
            }

            std::size_t rows = root["A"].arr_val.size();
            std::size_t cols = (rows > 0) ? root["A"].arr_val[0].arr_val.size() : 0;
            milp.A = vortex::Matrix<double>(rows, cols);
            std::size_t nnz = 0;
            for (std::size_t i = 0; i < rows; ++i) {
                for (std::size_t j = 0; j < cols; ++j) {
                    double v = root["A"].arr_val[i].arr_val[j].as_double(0.0);
                    milp.A(i, j) = v;
                    if (std::abs(v) > 1e-15) nnz++;
                }
            }

            for (const auto& item : root["b"].arr_val) {
                milp.b.push_back(item.as_double(0.0));
            }

            if (root["constraint_types"].is_array() && root["constraint_types"].arr_val.size() == rows) {
                for (const auto& ct : root["constraint_types"].arr_val) {
                    std::string s = ct.as_string("LEQ");
                    if (s == "GEQ") milp.constraint_types.push_back(vortex::ConstraintType::GEQ);
                    else if (s == "EQ") milp.constraint_types.push_back(vortex::ConstraintType::EQ);
                    else milp.constraint_types.push_back(vortex::ConstraintType::LEQ);
                }
            } else {
                milp.constraint_types.assign(rows, vortex::ConstraintType::LEQ);
            }

            // Variable types
            milp.variable_types.assign(cols, vortex::VariableType::CONTINUOUS);
            if (root["variable_types"].is_array() && root["variable_types"].arr_val.size() == cols) {
                for (std::size_t j = 0; j < cols; ++j) {
                    std::string vt = root["variable_types"].arr_val[j].as_string("CONTINUOUS");
                    if (vt == "INTEGER") milp.variable_types[j] = vortex::VariableType::INTEGER;
                    else if (vt == "BINARY") milp.variable_types[j] = vortex::VariableType::BINARY;
                }
            }

            // Variable bounds
            milp.bounds.assign(cols, vortex::VariableBounds{0.0, std::numeric_limits<double>::infinity()});
            if (root["bounds"].is_array() && root["bounds"].arr_val.size() == cols) {
                for (std::size_t j = 0; j < cols; ++j) {
                    const auto& bobj = root["bounds"].arr_val[j];
                    if (bobj.is_object()) {
                        milp.bounds[j].lower = bobj["lower"].as_double(0.0);
                        milp.bounds[j].upper = bobj["upper"].as_double(std::numeric_limits<double>::infinity());
                    }
                }
            }

            vortex::MILPOptions opts;
            if (root["options"].is_object()) {
                const auto& o = root["options"];
                if (o["node_limit"].is_number()) opts.node_limit = static_cast<std::size_t>(o["node_limit"].as_int());
                if (o["time_limit_sec"].is_number()) opts.time_limit_sec = o["time_limit_sec"].as_double();
            }

            vortex::MILPSolver solver(opts);
            vortex::MILPResult res = solver.solve(milp);

            auto t_end = std::chrono::high_resolution_clock::now();
            double wall_time_ms = std::chrono::duration<double, std::milli>(t_end - t_start).count();

            out_json << "  \"success\": true,\n";
            out_json << "  \"solver_type\": \"MILP\",\n";
            out_json << "  \"status\": " << escape_json_string(to_string(res.status)) << ",\n";
            out_json << "  \"message\": " << escape_json_string(res.message) << ",\n";
            out_json << "  \"objective_value\": " << format_double(res.objective_value) << ",\n";
            out_json << "  \"variable_values\": " << format_vector(res.variable_values) << ",\n";
            out_json << "  \"best_bound\": " << format_double(res.best_bound) << ",\n";
            out_json << "  \"absolute_gap\": " << format_double(res.absolute_gap) << ",\n";
            out_json << "  \"relative_gap\": " << format_double(res.relative_gap) << ",\n";
            out_json << "  \"nodes_explored\": " << res.nodes_explored << ",\n";
            out_json << "  \"nodes_pruned_infeasible\": " << res.nodes_pruned_infeasible << ",\n";
            out_json << "  \"nodes_pruned_bound\": " << res.nodes_pruned_bound << ",\n";
            out_json << "  \"nodes_pruned_integer\": " << res.nodes_pruned_integer << ",\n";
            out_json << "  \"wall_time_ms\": " << format_double(wall_time_ms) << ",\n";
            vortex::SolverDeviceDiagnostics dev_diag;
            dev_diag.requested_backend = requested_backend;
            dev_diag.executed_backend = vortex::ExecutedBackend::CPU;
            out_json << "  \"device_diagnostics\": " << format_device_diagnostics(dev_diag) << ",\n";
            out_json << "  \"matrix_info\": {\n";
            out_json << "    \"rows\": " << rows << ",\n";
            out_json << "    \"cols\": " << cols << ",\n";
            out_json << "    \"nnz\": " << nnz << ",\n";
            out_json << "    \"density\": " << format_double((rows * cols > 0) ? (static_cast<double>(nnz) / (rows * cols)) : 0.0) << "\n";
            out_json << "  }\n";
        }
        else if (solver_type == "QP") {
            // Convex Quadratic Program: Min (1/2) x^T Q x + c^T x s.t. A x <= b
            vortex::QuadraticProgram qp;

            // Read Q
            std::size_t n = root["Q"].arr_val.size();
            qp.Q = vortex::Matrix<double>(n, n);
            for (std::size_t i = 0; i < n; ++i) {
                for (std::size_t j = 0; j < n; ++j) {
                    qp.Q(i, j) = root["Q"].arr_val[i].arr_val[j].as_double(0.0);
                }
            }

            // Read c
            for (const auto& item : root["c"].arr_val) {
                qp.c.push_back(item.as_double(0.0));
            }

            // Read A
            std::size_t m = root["A"].arr_val.size();
            std::size_t cols = (m > 0) ? root["A"].arr_val[0].arr_val.size() : n;
            qp.A = vortex::Matrix<double>(m, cols);
            std::size_t nnz = 0;
            for (std::size_t i = 0; i < m; ++i) {
                for (std::size_t j = 0; j < cols; ++j) {
                    double v = root["A"].arr_val[i].arr_val[j].as_double(0.0);
                    qp.A(i, j) = v;
                    if (std::abs(v) > 1e-15) nnz++;
                }
            }

            // Read b
            for (const auto& item : root["b"].arr_val) {
                qp.b.push_back(item.as_double(0.0));
            }

            vortex::QPOptions opts;
            opts.backend = requested_backend;
            if (root["options"].is_object()) {
                const auto& o = root["options"];
                if (o["primal_tolerance"].is_number()) opts.primal_tolerance = o["primal_tolerance"].as_double();
                if (o["dual_tolerance"].is_number()) opts.dual_tolerance = o["dual_tolerance"].as_double();
                if (o["duality_gap_tolerance"].is_number()) opts.duality_gap_tolerance = o["duality_gap_tolerance"].as_double();
                if (o["max_iterations"].is_number()) opts.max_iterations = static_cast<std::size_t>(o["max_iterations"].as_int());
            }

            vortex::ConvexQPSolver solver(opts);
            vortex::QPResult res = solver.solve(qp);

            auto t_end = std::chrono::high_resolution_clock::now();
            double wall_time_ms = std::chrono::duration<double, std::milli>(t_end - t_start).count();

            std::string status_str = "NUMERICAL_ERROR";
            switch (res.status) {
                case vortex::QPSolverStatus::OPTIMAL: status_str = "OPTIMAL"; break;
                case vortex::QPSolverStatus::MAX_ITERATIONS: status_str = "MAX_ITERATIONS"; break;
                case vortex::QPSolverStatus::NUMERICAL_ERROR: status_str = "NUMERICAL_ERROR"; break;
                case vortex::QPSolverStatus::UNBOUNDED: status_str = "UNBOUNDED"; break;
                case vortex::QPSolverStatus::INFEASIBLE: status_str = "INFEASIBLE"; break;
                case vortex::QPSolverStatus::INVALID_INPUT: status_str = "INVALID_INPUT"; break;
                case vortex::QPSolverStatus::UNSUPPORTED: status_str = "UNSUPPORTED"; break;
            }

            out_json << "  \"success\": true,\n";
            out_json << "  \"solver_type\": \"QP\",\n";
            out_json << "  \"status\": " << escape_json_string(status_str) << ",\n";
            out_json << "  \"message\": " << escape_json_string(res.message) << ",\n";
            out_json << "  \"objective_value\": " << format_double(res.objective_value) << ",\n";
            out_json << "  \"variable_values\": " << format_vector(res.x) << ",\n";
            out_json << "  \"lambda\": " << format_vector(res.lambda) << ",\n";
            out_json << "  \"slack\": " << format_vector(res.slack) << ",\n";
            out_json << "  \"iterations\": " << res.iterations << ",\n";
            out_json << "  \"primal_residual\": " << format_double(res.primal_residual) << ",\n";
            out_json << "  \"dual_residual\": " << format_double(res.dual_residual) << ",\n";
            out_json << "  \"duality_gap\": " << format_double(res.duality_gap) << ",\n";
            out_json << "  \"wall_time_ms\": " << format_double(wall_time_ms) << ",\n";
            out_json << "  \"device_diagnostics\": " << format_device_diagnostics(res.device_diagnostics) << ",\n";
            out_json << "  \"iteration_history\": [\n";
            for (std::size_t h_idx = 0; h_idx < res.iteration_history.size(); ++h_idx) {
                const auto& rec = res.iteration_history[h_idx];
                out_json << "    {\n"
                         << "      \"iteration\": " << rec.iteration << ",\n"
                         << "      \"objective\": " << format_double(rec.objective_value) << ",\n"
                         << "      \"primal_residual\": " << format_double(rec.primal_residual) << ",\n"
                         << "      \"dual_residual\": " << format_double(rec.dual_residual) << ",\n"
                         << "      \"duality_gap\": " << format_double(rec.duality_gap) << "\n"
                         << "    }" << (h_idx + 1 < res.iteration_history.size() ? "," : "") << "\n";
            }
            out_json << "  ],\n";
            out_json << "  \"matrix_info\": {\n";
            out_json << "    \"q_dim\": " << n << ",\n";
            out_json << "    \"constraints_m\": " << m << ",\n";
            out_json << "    \"constraints_n\": " << cols << ",\n";
            out_json << "    \"nnz\": " << nnz << ",\n";
            out_json << "    \"density\": " << format_double((m * cols > 0) ? (static_cast<double>(nnz) / (m * cols)) : 0.0) << "\n";
            out_json << "  }\n";
        } else {
            throw std::runtime_error("Unsupported solver type: " + solver_type);
        }

        out_json << "}\n";

        if (output_path == "-" || output_path == "stdout") {
            std::cout << out_json.str();
        } else {
            std::ofstream out(output_path);
            if (!out) {
                std::cerr << "Failed to write to output file: " << output_path << "\n";
                return 1;
            }
            out << out_json.str();
        }
        return 0;

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n";
        std::string out_path;
        for (int i = 1; i < argc; ++i) {
            std::string arg = argv[i];
            if ((arg == "--output" || arg == "-o") && i + 1 < argc) {
                out_path = argv[i + 1];
                break;
            } else if (arg == "--stdout") {
                out_path = "-";
                break;
            }
        }
        std::string err_json = "{\n  \"success\": false,\n  \"status\": \"INVALID_INPUT\",\n  \"message\": " + escape_json_string(e.what()) + "\n}\n";
        if (out_path == "-" || out_path == "stdout") {
            std::cout << err_json;
        } else if (!out_path.empty()) {
            std::ofstream out(out_path);
            if (out) {
                out << err_json;
            }
        }
        return 1;
    }
}
