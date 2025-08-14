-- crystal_bridge.lua
-- Lua script to expose Pokémon Crystal game state and receive actions from Python RL agent

local json = require('json')

local CrystalBridge = {}

-- Game state memory addresses (these need to be updated for Pokémon Crystal)
local MEMORY_ADDRESSES = {
    player_x = 0xDCB8,
    player_y = 0xDCB9,
    player_map = 0xDCB5,
    player_hp = 0xDCDA,
    player_max_hp = 0xDCDB,
    player_level = 0xDCD3,
    player_exp = 0xDCD5,
    money = 0xD84E,
    badges = 0xD855,
    party_count = 0xDCD7,
    -- Add more addresses as needed
}

-- Initialize the bridge
function CrystalBridge.init()
    print("Pokémon Crystal RL Bridge initialized")
end

-- Extract current game state
function CrystalBridge.get_state()
    local state = {}
    
    -- Read memory values
    state.player_x = memory.read_u8(MEMORY_ADDRESSES.player_x)
    state.player_y = memory.read_u8(MEMORY_ADDRESSES.player_y)
    state.player_map = memory.read_u8(MEMORY_ADDRESSES.player_map)
    state.player_hp = memory.read_u16_le(MEMORY_ADDRESSES.player_hp)
    state.player_max_hp = memory.read_u16_le(MEMORY_ADDRESSES.player_max_hp)
    state.player_level = memory.read_u8(MEMORY_ADDRESSES.player_level)
    state.player_exp = memory.read_u24_le(MEMORY_ADDRESSES.player_exp)
    state.money = memory.read_u24_le(MEMORY_ADDRESSES.money)
    state.badges = memory.read_u8(MEMORY_ADDRESSES.badges)
    state.party_count = memory.read_u8(MEMORY_ADDRESSES.party_count)
    
    return state
end

-- Execute action received from Python agent
function CrystalBridge.execute_action(action)
    -- Map action numbers to button presses
    local actions = {
        [0] = {},  -- No action
        [1] = {"Up"},
        [2] = {"Down"},
        [3] = {"Left"},
        [4] = {"Right"},
        [5] = {"A"},
        [6] = {"B"},
        [7] = {"Start"},
        [8] = {"Select"},
    }
    
    local buttons = actions[action] or {}
    
    -- Press buttons for one frame
    for _, button in ipairs(buttons) do
        joypad.set({[button] = true})
    end
end

-- Main communication loop
function CrystalBridge.step()
    local state = CrystalBridge.get_state()
    local state_json = json.encode(state)
    
    -- Write state to file for Python to read
    local file = io.open("state.json", "w")
    if file then
        file:write(state_json)
        file:close()
    end
    
    -- Read action from file
    local action_file = io.open("action.txt", "r")
    if action_file then
        local action = tonumber(action_file:read("*all"))
        action_file:close()
        
        if action then
            CrystalBridge.execute_action(action)
        end
        
        -- Remove action file to signal action was processed
        os.remove("action.txt")
    end
end

-- Register event handlers
event.onexit(function()
    -- Clean up temporary files
    os.remove("state.json")
    os.remove("action.txt")
end)

-- Initialize and run
CrystalBridge.init()

-- Main loop - call step every frame
while true do
    emu.frameadvance()
    CrystalBridge.step()
end
