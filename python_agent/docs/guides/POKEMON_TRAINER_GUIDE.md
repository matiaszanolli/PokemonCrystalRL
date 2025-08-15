# Pokemon Crystal Unified Trainer Guide

## 🎉 **One Script to Rule Them All!**

We've consolidated **18 different training scripts** into one powerful, unified system:

**`pokemon_trainer.py`** - Your single entry point for all Pokemon Crystal RL training modes.

---

## 🚀 **Quick Start**

### **Basic Usage:**
```bash
# Fast local training with web interface
python pokemon_trainer.py --rom ../roms/pokemon_crystal.gbc --mode fast_local --actions 1000 --web

# Ultra-fast rule-based training 
python pokemon_trainer.py --rom ../roms/pokemon_crystal.gbc --mode ultra_fast --actions 5000 --no-llm

# Progressive curriculum training
python pokemon_trainer.py --rom ../roms/pokemon_crystal.gbc --mode curriculum --episodes 50
```

### **Get Help:**
```bash
python pokemon_trainer.py --help
```

---

## 🎯 **Training Modes**

### **1. Fast Local (`fast_local`)** - *Recommended for most users*
- **Best for**: Smooth gameplay footage and real-time monitoring
- **Performance**: ~40 actions/sec with LLM intelligence
- **Features**: Direct PyBoy access, optimized screen capture, web interface
- **Use case**: Content creation, development, balanced training

```bash
python pokemon_trainer.py --rom game.gbc --mode fast_local --actions 1000 --web
```

### **2. Ultra Fast (`ultra_fast`)** - *Maximum Speed*
- **Best for**: Performance testing, benchmarking, rapid exploration
- **Performance**: 600+ actions/sec (rule-based, no LLM)
- **Features**: Pattern-based actions, minimal overhead
- **Use case**: Speed testing, system validation

```bash
python pokemon_trainer.py --rom game.gbc --mode ultra_fast --actions 10000 --no-llm
```

### **3. Curriculum (`curriculum`)** - *Progressive Learning*
- **Best for**: Systematic skill development, research
- **Performance**: Variable (depends on LLM interval)
- **Features**: 5-stage progression, mastery validation, knowledge transfer
- **Use case**: Educational content, structured learning

```bash
python pokemon_trainer.py --rom game.gbc --mode curriculum --episodes 100
```

### **4. Monitored (`monitored`)** - *Full Analysis*
- **Best for**: Detailed analysis, research, debugging
- **Performance**: Slower but comprehensive
- **Features**: Full environment wrapper, detailed logging, enhanced agent
- **Use case**: Research, debugging, comprehensive analysis

```bash
python pokemon_trainer.py --rom game.gbc --mode monitored --episodes 20 --web
```

---

## 🤖 **LLM Models**

### **Available Models:**
- **`smollm2:1.7b`** ⭐ *Recommended* - Ultra-fast, optimized for Pokemon RL
- **`llama3.2:1b`** - Fastest Llama model
- **`llama3.2:3b`** - Balanced speed vs quality
- **`qwen2.5:3b`** - Alternative fast option
- **`--no-llm`** - Rule-based only (maximum speed)

### **Model Comparison:**
| Model | Speed | Intelligence | Memory | Best For |
|-------|--------|-------------|---------|----------|
| SmolLM2-1.7B | ⚡⚡⚡ | ⭐⭐⭐ | 2GB | **Recommended** |
| Llama3.2-1B | ⚡⚡ | ⭐⭐ | 1GB | Ultra-fast |
| Llama3.2-3B | ⚡ | ⭐⭐⭐⭐ | 3GB | Quality |
| Rule-based | ⚡⚡⚡⚡ | ⭐ | 0MB | Speed testing |

---

## 📊 **Performance Benchmarks**

Based on our testing:

| Mode | Speed (actions/sec) | LLM Model | Use Case |
|------|-------------------|-----------|-----------|
| `ultra_fast --no-llm` | **630+** | None | Speed testing |
| `fast_local` | **40** | SmolLM2-1.7B | Content creation |
| `curriculum` | **20-30** | SmolLM2-1.7B | Learning |
| `monitored` | **5-15** | Enhanced Agent | Research |

---

## 🌐 **Web Interface**

Add `--web` to any mode for real-time monitoring:

```bash
python pokemon_trainer.py --rom game.gbc --mode fast_local --web --port 8080
```

**Features:**
- 📸 **Live gameplay screen** (10 FPS)
- 📊 **Real-time statistics** (actions/sec, episodes, LLM calls)
- 🎯 **Training progress** tracking
- 📈 **Performance graphs** and metrics

**Access:** `http://localhost:8080`

---

## ⚙️ **Configuration Options**

### **Core Options:**
```bash
--rom PATH          # Pokemon Crystal ROM file (required)
--mode MODE         # Training mode (fast_local, ultra_fast, curriculum, monitored)
--model MODEL       # LLM model (smollm2:1.7b, llama3.2:1b, etc.)
--actions N         # Maximum actions to perform
--episodes N        # Maximum episodes to run
```

### **Performance:**
```bash
--llm-interval N    # Actions between LLM calls (default: 10)
--no-llm           # Disable LLM, use rule-based actions
--no-capture       # Disable screen capture for max speed
```

### **Interface:**
```bash
--web              # Enable web interface
--port N           # Web interface port (default: 8080)
--windowed         # Show game window (instead of headless)
```

### **Advanced:**
```bash
--save-state PATH  # Load from save state file
--debug            # Enable debug mode
```

---

## 📚 **Examples**

### **Content Creation Setup:**
```bash
# Perfect for recording gameplay footage
python pokemon_trainer.py \
  --rom ../roms/pokemon_crystal.gbc \
  --mode fast_local \
  --model smollm2:1.7b \
  --actions 2000 \
  --web \
  --windowed
```

### **Speed Benchmarking:**
```bash
# Maximum performance testing
python pokemon_trainer.py \
  --rom ../roms/pokemon_crystal.gbc \
  --mode ultra_fast \
  --actions 10000 \
  --no-llm \
  --no-capture
```

### **Research Setup:**
```bash
# Comprehensive analysis and monitoring
python pokemon_trainer.py \
  --rom ../roms/pokemon_crystal.gbc \
  --mode monitored \
  --model llama3.2:3b \
  --episodes 50 \
  --web \
  --debug
```

### **Curriculum Learning:**
```bash
# Progressive skill development
python pokemon_trainer.py \
  --rom ../roms/pokemon_crystal.gbc \
  --mode curriculum \
  --model smollm2:1.7b \
  --episodes 100 \
  --llm-interval 15 \
  --web
```

---

## 📁 **Output Files**

### **Generated Files:**
- **`training_stats.json`** - Performance statistics and metrics
- **`curriculum_validation.db`** - Stage validation data (curriculum mode)
- **Various logs and debug files** - Depending on mode and settings

### **Stats File Example:**
```json
{
  "start_time": 1699999999.0,
  "total_actions": 1000,
  "total_episodes": 10,
  "llm_calls": 100,
  "actions_per_second": 38.6,
  "mode": "fast_local",
  "model": "smollm2:1.7b",
  "end_time": 1699999999.0
}
```

---

## 🏆 **Best Practices**

### **For Content Creation:**
1. Use `fast_local` mode with `--web --windowed`
2. Choose `smollm2:1.7b` for best speed/quality balance
3. Set appropriate `--actions` limit for your content length
4. Monitor via web interface for real-time feedback

### **For Research:**
1. Use `monitored` mode with comprehensive logging
2. Choose higher-quality models for better decision analysis
3. Enable `--debug` for detailed insights
4. Save and analyze the generated statistics

### **For Performance Testing:**
1. Use `ultra_fast` with `--no-llm` for maximum speed
2. Disable capture with `--no-capture`
3. Use large `--actions` values (5000+)
4. Compare performance across different configurations

---

## 🔧 **Troubleshooting**

### **Common Issues:**

**1. "PyBoy not available"**
```bash
pip install pyboy pillow numpy
```

**2. "Model not found"**
- Models auto-download on first use
- Ensure Ollama is running: `ollama serve`

**3. "ROM file not found"**
- Check ROM path: `--rom ../roms/pokemon_crystal.gbc`
- Ensure file exists and is readable

**4. Slow performance**
- Try `smollm2:1.7b` model (fastest)
- Increase `--llm-interval` for fewer LLM calls
- Use `--no-capture` to disable screen capture
- Try `ultra_fast` mode with `--no-llm`

**5. Web interface not loading**
- Check if port is available: `--port 8081`
- Ensure firewall allows local connections
- Try different browser or incognito mode

---

## 🎯 **What's Next?**

The unified trainer gives you everything in one script:

✅ **Consolidated**: 18 scripts → 1 unified system  
✅ **Flexible**: Multiple modes and configurations  
✅ **Fast**: Up to 630+ actions/sec performance  
✅ **Intelligent**: SmolLM2-1.7B integration  
✅ **Visual**: Real-time web monitoring  
✅ **Research-ready**: Comprehensive statistics and logging  

**Ready to train Pokemon Crystal like never before!** 🚀🎮
