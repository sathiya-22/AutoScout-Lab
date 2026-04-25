```python
import redis
import json
import asyncio
from typing import Callable, Any, Dict, List, Optional

class MessageQueue:
    def __init__(self, host: str = 'localhost', port: int = 6379, db: int = 0, password: Optional[str] = None):
        self.redis_client = redis.Redis(host=host, port=port, db=db, password=password, decode_responses=True)
        self.connected = False
        self._ensure_connection()

    def _ensure_connection(self):
        try:
            self.redis_client.ping()
            self.connected = True
        except redis.exceptions.ConnectionError as e:
            self.connected = False
            raise ConnectionError(f"Failed to connect to Redis MessageQueue: {e}")

    async def publish(self, topic: str, message: Dict[str, Any]) -> str:
        if not self.connected:
            raise ConnectionError("MessageQueue is not connected to Redis.")
        
        try:
            message_id = self.redis_client.xadd(topic, {'payload': json.dumps(message)})
            return message_id.decode('utf-8') if isinstance(message_id, bytes) else message_id
        except Exception as e:
            raise RuntimeError(f"Failed to publish message to topic '{topic}': {e}")

    async def subscribe(self, topic: str, group_name: str, consumer_name: str, callback: Callable[[Dict[str, Any]], Any], block_ms: int = 5000, count: int = 10):
        if not self.connected:
            raise ConnectionError("MessageQueue is not connected to Redis.")

        try:
            self.redis_client.xgroup_create(topic, group_name, id='0', mkstream=True)
        except redis.exceptions.ResponseError as e:
            if "BUSYGROUP" not in str(e): 
                raise RuntimeError(f"Failed to create consumer group '{group_name}' for topic '{topic}': {e}")
        
        while True:
            try:
                response = self.redis_client.xreadgroup(
                    groupname=group_name,
                    consumername=consumer_name,
                    streams={topic: '>'},
                    count=count,
                    block=block_ms
                )
                
                if response:
                    for stream_name, messages in response:
                        for message_id, message_data in messages:
                            try:
                                payload_str = message_data.get('payload')
                                if payload_str:
                                    message_payload = json.loads(payload_str)
                                    message_payload['__message_id__'] = message_id.decode('utf-8')
                                    await self._run_callback(callback, message_payload)
                                    self.redis_client.xack(topic, group_name, message_id)
                                else:
                                    self.redis_client.xack(topic, group_name, message_id)
                            except json.JSONDecodeError:
                                self.redis_client.xack(topic, group_name, message_id)
                            except Exception:
                                pass # Message remains pending for retry or other consumer

            except asyncio.CancelledError:
                break
            except ConnectionError as ce:
                await asyncio.sleep(1)
                try:
                    self._ensure_connection()
                except ConnectionError:
                    pass
            except Exception:
                await asyncio.sleep(1)

    async def _run_callback(self, callback: Callable[[Dict[str, Any]], Any], message: Dict[str, Any]):
        if asyncio.iscoroutinefunction(callback):
            await callback(message)
        else:
            callback(message)

    def close(self):
        if self.connected:
            self.redis_client.close()
            self.connected = False
```