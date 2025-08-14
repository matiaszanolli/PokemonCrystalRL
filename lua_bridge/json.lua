-- json.lua
-- Lightweight JSON encoder/decoder for Lua
-- Based on dkjson-style implementation

local json = {}

local function escape_str(s)
    local in_char  = {'\\', '"', '/', '\b', '\f', '\n', '\r', '\t'}
    local out_char = {'\\', '"', '/',  'b',  'f',  'n',  'r',  't'}
    for i, c in ipairs(in_char) do
        s = s:gsub(c, '\\' .. out_char[i])
    end
    return s
end

local function encode(val, stack)
    local t = type(val)
    
    if t == 'nil' then
        return 'null'
    elseif t == 'boolean' then
        return tostring(val)
    elseif t == 'number' then
        return tostring(val)
    elseif t == 'string' then
        return '"' .. escape_str(val) .. '"'
    elseif t == 'table' then
        -- Check for circular references
        stack = stack or {}
        if stack[val] then error("circular reference") end
        stack[val] = true
        
        local result = {}
        local is_array = true
        local n = 0
        
        -- Check if table is array-like
        for k, v in pairs(val) do
            n = n + 1
            if type(k) ~= 'number' or k ~= n then
                is_array = false
                break
            end
        end
        
        if is_array and n > 0 then
            -- Encode as array
            for i = 1, n do
                result[i] = encode(val[i], stack)
            end
            stack[val] = nil
            return '[' .. table.concat(result, ',') .. ']'
        else
            -- Encode as object
            for k, v in pairs(val) do
                if type(k) ~= 'string' then
                    k = tostring(k)
                end
                table.insert(result, '"' .. escape_str(k) .. '":' .. encode(v, stack))
            end
            stack[val] = nil
            return '{' .. table.concat(result, ',') .. '}'
        end
    else
        error("cannot encode " .. t)
    end
end

local function skip_whitespace(str, pos)
    while pos <= #str and str:sub(pos, pos):match('%s') do
        pos = pos + 1
    end
    return pos
end

local function decode_string(str, pos)
    local result = ''
    pos = pos + 1  -- skip opening quote
    
    while pos <= #str do
        local c = str:sub(pos, pos)
        if c == '"' then
            return result, pos + 1
        elseif c == '\\' then
            pos = pos + 1
            c = str:sub(pos, pos)
            if c == '"' then result = result .. '"'
            elseif c == '\\' then result = result .. '\\'
            elseif c == '/' then result = result .. '/'
            elseif c == 'b' then result = result .. '\b'
            elseif c == 'f' then result = result .. '\f'
            elseif c == 'n' then result = result .. '\n'
            elseif c == 'r' then result = result .. '\r'
            elseif c == 't' then result = result .. '\t'
            else result = result .. c end
        else
            result = result .. c
        end
        pos = pos + 1
    end
    error("unterminated string")
end

local function decode_number(str, pos)
    local start = pos
    if str:sub(pos, pos) == '-' then pos = pos + 1 end
    
    while pos <= #str and str:sub(pos, pos):match('%d') do
        pos = pos + 1
    end
    
    if pos <= #str and str:sub(pos, pos) == '.' then
        pos = pos + 1
        while pos <= #str and str:sub(pos, pos):match('%d') do
            pos = pos + 1
        end
    end
    
    if pos <= #str and str:sub(pos, pos):match('[eE]') then
        pos = pos + 1
        if pos <= #str and str:sub(pos, pos):match('[+-]') then
            pos = pos + 1
        end
        while pos <= #str and str:sub(pos, pos):match('%d') do
            pos = pos + 1
        end
    end
    
    return tonumber(str:sub(start, pos - 1)), pos
end

local decode_value

local function decode_array(str, pos)
    local result = {}
    pos = pos + 1  -- skip '['
    pos = skip_whitespace(str, pos)
    
    if pos <= #str and str:sub(pos, pos) == ']' then
        return result, pos + 1
    end
    
    while true do
        local value
        value, pos = decode_value(str, pos)
        table.insert(result, value)
        pos = skip_whitespace(str, pos)
        
        if pos <= #str and str:sub(pos, pos) == ']' then
            return result, pos + 1
        elseif pos <= #str and str:sub(pos, pos) == ',' then
            pos = skip_whitespace(str, pos + 1)
        else
            error("expected ',' or ']'")
        end
    end
end

local function decode_object(str, pos)
    local result = {}
    pos = pos + 1  -- skip '{'
    pos = skip_whitespace(str, pos)
    
    if pos <= #str and str:sub(pos, pos) == '}' then
        return result, pos + 1
    end
    
    while true do
        pos = skip_whitespace(str, pos)
        if pos > #str or str:sub(pos, pos) ~= '"' then
            error("expected string key")
        end
        
        local key
        key, pos = decode_string(str, pos)
        pos = skip_whitespace(str, pos)
        
        if pos > #str or str:sub(pos, pos) ~= ':' then
            error("expected ':'")
        end
        pos = skip_whitespace(str, pos + 1)
        
        local value
        value, pos = decode_value(str, pos)
        result[key] = value
        pos = skip_whitespace(str, pos)
        
        if pos <= #str and str:sub(pos, pos) == '}' then
            return result, pos + 1
        elseif pos <= #str and str:sub(pos, pos) == ',' then
            pos = skip_whitespace(str, pos + 1)
        else
            error("expected ',' or '}'")
        end
    end
end

decode_value = function(str, pos)
    pos = skip_whitespace(str, pos)
    if pos > #str then error("unexpected end of input") end
    
    local c = str:sub(pos, pos)
    if c == '"' then
        return decode_string(str, pos)
    elseif c:match('[%-0-9]') then
        return decode_number(str, pos)
    elseif c == '[' then
        return decode_array(str, pos)
    elseif c == '{' then
        return decode_object(str, pos)
    elseif str:sub(pos, pos + 3) == 'true' then
        return true, pos + 4
    elseif str:sub(pos, pos + 4) == 'false' then
        return false, pos + 5
    elseif str:sub(pos, pos + 3) == 'null' then
        return nil, pos + 4
    else
        error("unexpected character: " .. c)
    end
end

function json.encode(val)
    return encode(val)
end

function json.decode(str)
    local result, pos = decode_value(str, 1)
    pos = skip_whitespace(str, pos)
    if pos <= #str then
        error("trailing garbage")
    end
    return result
end

return json
