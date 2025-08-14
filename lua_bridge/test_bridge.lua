-- test_bridge.lua
-- Simplified test script to debug Lua-Python communication

print("Test Lua script started")

-- Simple JSON encoder for basic types
local function simple_json_encode(data)
    if type(data) == "table" then
        local parts = {}
        for k, v in pairs(data) do
            local key = string.format('"%s"', tostring(k))
            local value
            if type(v) == "string" then
                value = string.format('"%s"', v)
            elseif type(v) == "number" then
                value = tostring(v)
            else
                value = string.format('"%s"', tostring(v))
            end
            table.insert(parts, key .. ":" .. value)
        end
        return "{" .. table.concat(parts, ",") .. "}"
    else
        return string.format('"%s"', tostring(data))
    end
end

-- Test state data
local test_state = {
    player_x = 10,
    player_y = 15,
    player_hp = 100,
    test_message = "Hello from Lua!",
    frame_count = 0
}

-- Initialize frame counter
local frame_count = 0

print("About to enter main loop")

-- Main loop
while true do
    frame_count = frame_count + 1
    test_state.frame_count = frame_count
    
    -- Write state every 60 frames (roughly once per second)
    if frame_count % 60 == 0 then
        local state_json = simple_json_encode(test_state)
        
        -- Try to write state file
        local success, err = pcall(function()
            local file = io.open("state.json", "w")
            if file then
                file:write(state_json)
                file:close()
                print("State file written:", state_json)
            else
                print("Failed to open state.json for writing")
            end
        end)
        
        if not success then
            print("Error writing state file:", err)
        end
    end
    
    -- Try to read action file
    local success, err = pcall(function()
        local action_file = io.open("action.txt", "r")
        if action_file then
            local action = action_file:read("*all")
            action_file:close()
            print("Action received:", action)
            
            -- Remove action file
            os.remove("action.txt")
            print("Action file processed and removed")
        end
    end)
    
    if not success then
        print("Error reading action file:", err)
    end
    
    -- Advance frame
    emu.frameadvance()
    
    -- Print status every 300 frames (every 5 seconds)
    if frame_count % 300 == 0 then
        print("Frame", frame_count, "- Test script running")
    end
end
