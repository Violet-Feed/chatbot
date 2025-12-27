-- KEYS[1] = bucket hash
-- ARGV = now_ms, cap, period_ms, refund
local key = KEYS[1]
local now = tonumber(ARGV[1])
local cap = tonumber(ARGV[2])
local period = tonumber(ARGV[3])
local refund = tonumber(ARGV[4])

local tokens = tonumber(redis.call("HGET", key, "tokens") or tostring(cap))
local last = tonumber(redis.call("HGET", key, "last_refill_ms") or tostring(now))

local elapsed = now - last
if elapsed >= period then
  tokens = cap
  last = now
end

tokens = math.min(cap, tokens + refund)
redis.call("HSET", key, "tokens", tokens)
redis.call("HSET", key, "last_refill_ms", last)
redis.call("PEXPIRE", key, period * 2)
return tokens
