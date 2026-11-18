#!/usr/bin/env python3
"""
测试MMM3地图配置脚本
用于验证地图文件是否正确注册和配置
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

def test_map_registration():
    """测试地图注册"""
    print("=" * 60)
    print("测试地图注册")
    print("=" * 60)
    
    try:
        from envs.starcraft.smac_maps import get_map_params
        
        # 测试MMM3配置
        print("\n1. 检查MMM3地图配置...")
        try:
            params = get_map_params("MMM3")
            print(f"✅ MMM3地图配置找到:")
            print(f"   - n_agents: {params['n_agents']}")
            print(f"   - n_enemies: {params['n_enemies']}")
            print(f"   - limit: {params['limit']}")
            print(f"   - a_race: {params['a_race']}")
            print(f"   - b_race: {params['b_race']}")
            print(f"   - unit_type_bits: {params['unit_type_bits']}")
            print(f"   - map_type: {params['map_type']}")
        except KeyError as e:
            print(f"❌ MMM3地图配置未找到: {e}")
            return False
        
        # 测试MMM2配置（对比）
        print("\n2. 检查MMM2地图配置（对比）...")
        try:
            params_mmm2 = get_map_params("MMM2")
            print(f"✅ MMM2地图配置:")
            print(f"   - n_agents: {params_mmm2['n_agents']}")
            print(f"   - n_enemies: {params_mmm2['n_enemies']}")
            print(f"   - limit: {params_mmm2['limit']}")
            print(f"   - map_type: {params_mmm2['map_type']}")
        except KeyError:
            print("⚠️  MMM2地图配置未找到（可能使用smac库默认配置）")
        
        return True
    except Exception as e:
        print(f"❌ 测试地图注册失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_map_file_path():
    """测试地图文件路径"""
    print("\n" + "=" * 60)
    print("测试地图文件路径")
    print("=" * 60)
    
    map_file = "smac_maps/MMM3.SC2Map"
    print(f"\n1. 检查地图文件: {map_file}")
    
    if os.path.exists(map_file):
        file_size = os.path.getsize(map_file)
        print(f"✅ 地图文件存在")
        print(f"   - 路径: {os.path.abspath(map_file)}")
        print(f"   - 大小: {file_size / 1024:.2f} KB")
        
        # 检查文件权限
        if os.access(map_file, os.R_OK):
            print(f"   - 权限: 可读 ✅")
        else:
            print(f"   - 权限: 不可读 ❌")
    else:
        print(f"❌ 地图文件不存在: {map_file}")
        return False
    
    # 检查smac地图目录
    print(f"\n2. 检查smac_maps目录...")
    if os.path.isdir("smac_maps"):
        files = [f for f in os.listdir("smac_maps") if f.endswith(".SC2Map")]
        print(f"✅ smac_maps目录存在")
        print(f"   - 地图文件数量: {len(files)}")
        print(f"   - MMM相关地图: {[f for f in files if 'MMM' in f]}")
    else:
        print(f"❌ smac_maps目录不存在")
        return False
    
    return True


def test_map_in_pysc2():
    """测试地图是否在pysc2中注册"""
    print("\n" + "=" * 60)
    print("测试pysc2地图注册")
    print("=" * 60)
    
    try:
        from pysc2 import maps
        
        print("\n1. 尝试获取MMM3地图...")
        try:
            mmm3_map = maps.get("MMM3")
            print(f"✅ MMM3地图在pysc2中找到")
            print(f"   - 路径: {mmm3_map.path}")
            print(f"   - 文件名: {mmm3_map.filename}")
        except KeyError:
            print("❌ MMM3地图未在pysc2中注册")
            print("\n尝试检查pysc2地图注册...")
            try:
                all_maps = maps.get_maps()
                mmm_maps = [m for m in all_maps if 'MMM' in m]
                print(f"   找到的MMM相关地图: {[m.filename for m in mmm_maps]}")
            except Exception as e:
                print(f"   ⚠️  无法获取地图列表: {e}")
            return False
        except Exception as e:
            print(f"❌ 获取MMM3地图失败: {e}")
            import traceback
            traceback.print_exc()
            return False
        
        return True
    except ImportError as e:
        print(f"⚠️  无法导入pysc2.maps: {e}")
        print("   这可能是正常的，如果smac使用自己的地图系统")
        return None


def test_environment_creation():
    """测试环境创建（不实际启动SC2）"""
    print("\n" + "=" * 60)
    print("测试环境创建")
    print("=" * 60)
    
    try:
        from envs.starcraft.StarCraft2Env import StarCraft2Env
        
        print("\n1. 尝试创建StarCraft2Env（不启动SC2）...")
        
        # 只检查配置是否正确解析
        try:
            env = StarCraft2Env(
                map_name="MMM3",
                difficulty="7",
                debug=False,
                heuristic_ai=False
            )
            
            print(f"✅ 环境对象创建成功")
            print(f"   - n_agents: {env.n_agents}")
            print(f"   - n_enemies: {env.n_enemies}")
            print(f"   - episode_limit: {env.episode_limit}")
            print(f"   - map_type: {env.map_type}")
            print(f"   - unit_type_bits: {env.unit_type_bits}")
            print(f"   - _agent_race: {env._agent_race}")
            print(f"   - _bot_race: {env._bot_race}")
            
            # 检查动作空间
            print(f"\n2. 检查动作空间...")
            print(f"   - n_actions: {env.n_actions}")
            print(f"   - n_actions_no_attack: {env.n_actions_no_attack}")
            print(f"   - n_actions_move: {env.n_actions_move}")
            print(f"   - 预期总动作数: {env.n_actions_no_attack + env.n_enemies} = {env.n_actions}")
            
            return True
        except Exception as e:
            print(f"❌ 环境创建失败: {e}")
            import traceback
            traceback.print_exc()
            return False
            
    except Exception as e:
        print(f"❌ 测试环境创建失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """主函数"""
    print("\n" + "🔍 MMM3地图配置测试脚本" + "\n")
    
    results = []
    
    # 1. 测试地图注册
    results.append(("地图注册", test_map_registration()))
    
    # 2. 测试地图文件路径
    results.append(("地图文件路径", test_map_file_path()))
    
    # 3. 测试pysc2地图注册
    pysc2_result = test_map_in_pysc2()
    if pysc2_result is not None:
        results.append(("pysc2地图注册", pysc2_result))
    
    # 4. 测试环境创建
    results.append(("环境创建", test_environment_creation()))
    
    # 总结
    print("\n" + "=" * 60)
    print("测试总结")
    print("=" * 60)
    
    for name, result in results:
        status = "✅ 通过" if result else "❌ 失败"
        if result is None:
            status = "⚠️  跳过"
        print(f"{name}: {status}")
    
    all_passed = all(r for r in results if r is not None)
    
    if all_passed:
        print("\n✅ 所有测试通过！地图配置看起来正确。")
        print("\n如果SC2仍然崩溃，可能的原因：")
        print("1. 地图文件本身的问题（需要检查SC2Map文件内容）")
        print("2. SC2版本兼容性问题")
        print("3. 资源限制（内存/CPU）")
    else:
        print("\n❌ 部分测试失败，请检查上述错误信息。")
    
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())

