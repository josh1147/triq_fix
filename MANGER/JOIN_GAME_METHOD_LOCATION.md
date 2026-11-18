# join_game 方法位置和调用链

## 📍 方法位置

### 1. 调用位置（你的代码）

**文件**: `src/envs/starcraft/StarCraft2Env.py`  
**行号**: 314  
**方法**: `_launch()`

```python:src/envs/starcraft/StarCraft2Env.py
def _launch(self):
    """Launch the StarCraft II game."""
    self._run_config = run_configs.get(version=self.game_version)
    _map = maps.get(self.map_name)

    # Setting up the interface
    interface_options = sc_pb.InterfaceOptions(raw=True, score=False)
    self._sc2_proc = self._run_config.start(window_size=self.window_size, want_rgb=False)
    self._controller = self._sc2_proc.controller  # pysc2的RemoteController实例

    # Request to create the game
    create = sc_pb.RequestCreateGame(
        local_map=sc_pb.LocalMap(
            map_path=_map.path,
            map_data=self._run_config.map_data(_map.path)),
        realtime=False,
        random_seed=self._seed)
    create.player_setup.add(type=sc_pb.Participant)
    create.player_setup.add(type=sc_pb.Computer, race=races[self._bot_race],
                            difficulty=difficulties[self.difficulty])
    self._controller.create_game(create)  # ✅ 成功执行

    join = sc_pb.RequestJoinGame(race=races[self._agent_race],
                                 options=interface_options)
    self._controller.join_game(join)  # ❌ 在这里崩溃
```

### 2. join_game 方法实现（pysc2库）

**文件**: `/root/autodl-tmp/miniconda3/envs/pymarl/lib/python3.8/site-packages/pysc2/lib/remote_controller.py`  
**行号**: 203-205

```python
@valid_status(Status.launched, Status.init_game)
@decorate_check_error(sc_pb.ResponseJoinGame.Error)
@sw.decorate
def join_game(self, req_join_game):
    """Join a game, done by all connected clients."""
    return self._client.send(join_game=req_join_game)
```

### 3. 底层发送实现（protocol.py）

**文件**: `/root/autodl-tmp/miniconda3/envs/pymarl/lib/python3.8/site-packages/pysc2/lib/protocol.py`

`self._client.send()` 方法会将请求发送给SC2进程，并等待响应。

## 🔗 调用链

```
StarCraft2Env._launch()
  └─> self._controller.join_game(join)
       └─> RemoteController.join_game(req_join_game)
            └─> self._client.send(join_game=req_join_game)
                 └─> Protocol.send()  [通过WebSocket发送给SC2]
                      └─> SC2进程处理请求
                           ❌ SC2进程崩溃 (Signal 11)
```

## 🔍 关键信息

### self._controller 的类型

```python
self._controller = self._sc2_proc.controller
```

- `_sc2_proc`: pysc2的SC2Process实例
- `controller`: RemoteController实例
- **类型**: `pysc2.lib.remote_controller.RemoteController`

### join_game 的装饰器

1. **@valid_status(Status.launched, Status.init_game)**
   - 确保SC2进程处于正确状态
   - 必须在`launched`或`init_game`状态

2. **@decorate_check_error(sc_pb.ResponseJoinGame.Error)**
   - 检查响应中的错误
   - 如果有错误会抛出RequestError

3. **@sw.decorate**
   - 性能统计装饰器

### 实际执行流程

```python
# 1. 验证状态
if self.status not in (Status.launched, Status.init_game):
    raise ProtocolError(...)

# 2. 发送请求
response = self._client.send(join_game=req_join_game)

# 3. 检查错误
if response.HasField("error"):
    raise RequestError(...)

# 4. 返回响应
return response
```

## 🚨 崩溃发生位置

**崩溃发生在SC2进程内部**，不是Python代码中：

```
1. Python代码调用 join_game
2. 请求通过WebSocket发送给SC2进程
3. SC2进程接收到join_game请求
4. SC2进程开始处理请求（初始化游戏状态）
5. ❌ SC2进程崩溃 (Signal 11)
6. WebSocket连接断开
7. Python检测到连接断开，抛出ConnectionError
```

## 💡 为什么在这里崩溃？

`join_game`阶段SC2需要：
1. ✅ 已经加载了地图文件（create_game成功）
2. ✅ 已经初始化了游戏状态
3. ❌ **在创建玩家和单位时崩溃**

**可能的原因**：
- MMM3地图文件中的单位配置与代码期望不匹配
- 地图触发器配置有问题
- 单位数量或类型不符合地图文件的实际情况

## 📝 总结

**join_game方法位置**：
- **调用**: `src/envs/starcraft/StarCraft2Env.py:314`
- **实现**: `pysc2/lib/remote_controller.py:203`
- **底层**: `pysc2/lib/protocol.py` (send方法)

**崩溃位置**: SC2进程内部（不是Python代码）

