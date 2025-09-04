"""
Pokemon Crystal Memory Reader
Provides direct access to game memory addresses for debugging and monitoring
"""

import logging
import time

logger = logging.getLogger(__name__)


class PokemonCrystalMemoryReader:
    """Read Pokemon Crystal game state from PyBoy memory"""
    
    MEMORY_ADDRESSES = {
        'PARTY_COUNT': 0xDCDE,
        'PLAYER_MAP': 0xDCE6,
        'PLAYER_X': 0xDCE7,
        'PLAYER_Y': 0xDCE8, 
        'MONEY': [0xD844, 0xD845, 0xD846],  # 3-byte BCD
        'BADGES': 0xD857,
        'IN_BATTLE': 0xD062,
        'PLAYER_LEVEL': 0xDD2F,
        'HP_CURRENT': [0xDD2E, 0xDD2F],  # 2-byte
        'HP_MAX': [0xDD30, 0xDD31],      # 2-byte
        'WRAM_BANK': 0xFF70,  # Current WRAM bank
        'STEP_COUNTER': [0xD4B0, 0xD4B1],  # 2-byte step counter
        'FACING_DIRECTION': 0xDCE9,
        'MAP_GROUP': 0xDCE5,
        'TIME_HOURS': 0xD4B2,
        'TIME_MINUTES': 0xD4B3,
        'PLAYER_NAME': list(range(0xD47D, 0xD487)),  # Player name (10 bytes)
    }
    
    def __init__(self, pyboy):
        self.pyboy = pyboy
        self.last_read_time = 0
        self.cached_state = {}
        self.cache_duration = 0.1  # Cache for 100ms to avoid excessive reads
        
    def read_game_state(self):
        """Read complete game state from memory"""
        current_time = time.time()
        
        # Use cached state if recent
        if current_time - self.last_read_time < self.cache_duration:
            return self.cached_state.copy()
            
        try:
            state = {
                'timestamp': current_time,
                'memory_read_success': True
            }
            
            for name, addr in self.MEMORY_ADDRESSES.items():
                try:
                    if isinstance(addr, list):
                        if name == 'MONEY':
                            state[name] = self._read_bcd_money(addr)
                        elif name == 'PLAYER_NAME':
                            state[name] = self._read_player_name(addr)
                        else:
                            state[name] = self._read_multi_byte(addr)
                    else:
                        state[name] = self.pyboy.memory[addr]
                except Exception as e:
                    logger.debug(f"Error reading {name}: {e}")
                    state[name] = None
            
            # Calculate derived values
            state['COORDS'] = [state.get('PLAYER_X', 0), state.get('PLAYER_Y', 0)]
            state['HP_PERCENTAGE'] = self._calculate_hp_percentage(
                state.get('HP_CURRENT', 0), 
                state.get('HP_MAX', 1)
            )
            state['HAS_POKEMON'] = state.get('PARTY_COUNT', 0) > 0
            
            # Cache the result
            self.cached_state = state
            self.last_read_time = current_time
            
            return state
            
        except Exception as e:
            logger.error(f"Memory read error: {e}")
            return {
                'timestamp': current_time,
                'memory_read_success': False,
                'error': str(e)
            }
    
    def _read_multi_byte(self, addresses):
        """Read multi-byte value (little-endian)"""
        value = 0
        for i, addr in enumerate(addresses):
            byte_val = self.pyboy.memory[addr]
            value |= (byte_val << (8 * i))
        return value
    
    def _read_bcd_money(self, addresses):
        """Read BCD-encoded money value"""
        try:
            # Pokemon Crystal stores money in BCD format
            bcd_bytes = [self.pyboy.memory[addr] for addr in addresses]
            
            # Convert BCD to decimal
            money = 0
            for i, bcd_byte in enumerate(reversed(bcd_bytes)):
                tens = (bcd_byte >> 4) & 0xF
                ones = bcd_byte & 0xF
                money = money * 100 + tens * 10 + ones
            
            # Sanity check - if money seems unreasonably high, it's probably corrupted
            if money > 999999:
                logger.debug(f"Unusually high money value: {money}, possible memory corruption")
                return 0
                
            return money
        except Exception as e:
            logger.debug(f"Error reading BCD money: {e}")
            return 0
    
    def _read_player_name(self, addresses):
        """Read player name from memory"""
        try:
            name_bytes = []
            for addr in addresses:
                byte_val = self.pyboy.memory[addr]
                if byte_val == 0x50:  # Terminator in Pokemon Crystal
                    break
                if byte_val != 0:
                    name_bytes.append(byte_val)
            
            # Convert to readable characters (simplified mapping)
            # Full character mapping would require Pokemon Crystal's character table
            readable_chars = []
            for byte_val in name_bytes:
                if 0xA1 <= byte_val <= 0xBA:  # A-Z range
                    readable_chars.append(chr(ord('A') + (byte_val - 0xA1)))
                elif 0xBB <= byte_val <= 0xD4:  # a-z range
                    readable_chars.append(chr(ord('a') + (byte_val - 0xBB)))
                elif 0xF6 <= byte_val <= 0xFF:  # 0-9 range  
                    readable_chars.append(chr(ord('0') + (byte_val - 0xF6)))
                else:
                    readable_chars.append('?')
            
            return ''.join(readable_chars) if readable_chars else "UNKNOWN"
            
        except Exception as e:
            logger.debug(f"Error reading player name: {e}")
            return "ERROR"
    
    def _calculate_hp_percentage(self, current_hp, max_hp):
        """Calculate HP percentage safely"""
        if max_hp == 0:
            return 0
        return min(100, max(0, int((current_hp / max_hp) * 100)))
    
    def get_debug_info(self):
        """Get debug information about memory reader state"""
        return {
            'cache_age_seconds': time.time() - self.last_read_time,
            'cache_duration': self.cache_duration,
            'pyboy_available': self.pyboy is not None,
            'memory_addresses_count': len(self.MEMORY_ADDRESSES),
        }