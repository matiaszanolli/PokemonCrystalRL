-- minimal_test.lua
-- Most basic test possible

print("=== MINIMAL LUA TEST STARTING ===")

-- Just try to create a file
local file = io.open("lua_test.txt", "w")
if file then
    file:write("Hello from Lua script!")
    file:close()
    print("=== FILE CREATED SUCCESSFULLY ===")
else
    print("=== FAILED TO CREATE FILE ===")
end

print("=== MINIMAL LUA TEST COMPLETE ===")

-- Don't run forever, just exit after a few frames
for i = 1, 60 do
    emu.frameadvance()
end

print("=== EXITING AFTER 60 FRAMES ===")
