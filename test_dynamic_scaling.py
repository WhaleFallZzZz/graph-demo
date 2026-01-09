#!/usr/bin/env python3
"""
测试动态资源分配系统

测试场景：
1. 单Worker活跃时，分配全部资源
2. 多Worker活跃时，平均分配资源
3. Worker状态同步和超时清理
4. 资源分配回调函数调用
"""

import os
import sys
import time
import json
from pathlib import Path

# 添加项目根目录到Python路径
current_dir = Path(__file__).parent
llama_dir = current_dir / "llama"
sys.path.insert(0, str(llama_dir))

from common.dynamic_resource_allocator import (
    DynamicScalingManager,
    ResourceAllocation,
    WorkerActivityMonitor,
    DynamicResourceAllocator
)

# 测试回调记录
allocation_history = []

def test_callback(allocation: ResourceAllocation):
    """测试回调函数"""
    allocation_history.append({
        'timestamp': time.time(),
        'allocation': allocation.to_dict()
    })
    print(f"📞 回调函数被调用: {allocation.to_dict()}")

def test_single_worker_scenario():
    """测试单Worker场景"""
    print("\n" + "="*60)
    print("测试场景1: 单Worker活跃")
    print("="*60)
    
    # 清理历史记录
    allocation_history.clear()
    
    # 创建管理器
    manager = DynamicScalingManager(
        worker_id="worker_1",
        total_workers=4,
        base_allocation=ResourceAllocation(
            max_concurrent_requests=10,
            rpm_limit=200,
            tpm_limit=10000,
            num_workers=10
        ),
        enable_scaling=True
    )
    
    # 设置回调
    manager.set_allocation_callback(test_callback)
    
    # 启动
    manager.start()
    
    # 标记worker_1为活跃
    manager.update_activity(is_active=True, current_load=0.8, active_tasks=5)
    
    # 等待调整
    time.sleep(12)
    
    # 获取状态
    status = manager.get_status()
    print(f"\n📊 当前状态:")
    print(f"  活跃Worker数: {status['active_workers']}/{status['total_workers']}")
    print(f"  当前分配: {status['current_allocation']}")
    print(f"  利用率: {status['utilization_ratio']}")
    
    # 验证：应该分配全部资源
    assert status['active_workers'] == 1, "应该只有1个活跃Worker"
    assert status['current_allocation']['max_concurrent_requests'] == 40, "应该分配全部并发数"
    assert status['current_allocation']['rpm_limit'] == 800, "应该分配全部RPM"
    
    print("\n✅ 单Worker场景测试通过")
    
    # 停止
    manager.stop()
    
    return True

def test_multiple_workers_scenario():
    """测试多Worker场景"""
    print("\n" + "="*60)
    print("测试场景2: 多Worker活跃")
    print("="*60)
    
    # 清理历史记录
    allocation_history.clear()
    
    # 创建4个Worker管理器
    managers = []
    for i in range(1, 5):
        manager = DynamicScalingManager(
            worker_id=f"worker_{i}",
            total_workers=4,
            base_allocation=ResourceAllocation(
                max_concurrent_requests=10,
                rpm_limit=200,
                tpm_limit=10000,
                num_workers=10
            ),
            enable_scaling=True
        )
        manager.set_allocation_callback(test_callback)
        managers.append(manager)
    
    # 启动所有Worker
    for manager in managers:
        manager.start()
    
    # 标记前3个Worker为活跃
    for i in range(1, 4):
        managers[i-1].update_activity(is_active=True, current_load=0.7, active_tasks=3)
    
    # 等待调整（增加等待时间以确保所有Worker状态都同步）
    time.sleep(20)
    
    # 获取状态
    status = managers[0].get_status()
    print(f"\n📊 当前状态:")
    print(f"  活跃Worker数: {status['active_workers']}/{status['total_workers']}")
    print(f"  当前分配: {status['current_allocation']}")
    print(f"  利用率: {status['utilization_ratio']}")
    
    # 验证：应该平均分配资源
    assert status['active_workers'] == 3, "应该有3个活跃Worker"
    assert status['current_allocation']['max_concurrent_requests'] == 13, "应该平均分配并发数 (40//3=13)"
    assert status['current_allocation']['rpm_limit'] == 266, "应该平均分配RPM (800//3=266)"
    
    print("\n✅ 多Worker场景测试通过")
    
    # 停止所有Worker
    for manager in managers:
        manager.stop()
    
    return True

def test_worker_timeout_scenario():
    """测试Worker超时场景"""
    print("\n" + "="*60)
    print("测试场景3: Worker超时清理")
    print("="*60)
    
    # 清理历史记录
    allocation_history.clear()
    
    # 创建管理器
    manager = DynamicScalingManager(
        worker_id="worker_1",
        total_workers=4,
        base_allocation=ResourceAllocation(
            max_concurrent_requests=10,
            rpm_limit=200,
            tpm_limit=10000,
            num_workers=10
        ),
        enable_scaling=True
    )
    
    manager.set_allocation_callback(test_callback)
    manager.start()
    
    # 标记为活跃
    manager.update_activity(is_active=True, current_load=0.5, active_tasks=2)
    
    # 等待调整
    time.sleep(12)
    
    status = manager.get_status()
    print(f"\n📊 活跃状态: {status['active_workers']}个活跃Worker")
    assert status['active_workers'] == 1, "应该有1个活跃Worker"
    
    # 停止更新，等待超时
    print("\n⏳ 停止更新，等待超时...")
    manager.update_activity(is_active=False, current_load=0.0, active_tasks=0)
    
    # 等待超时（默认30秒）
    time.sleep(35)
    
    status = manager.get_status()
    print(f"\n📊 超时后状态: {status['active_workers']}个活跃Worker")
    assert status['active_workers'] == 0, "超时后应该没有活跃Worker"
    
    print("\n✅ Worker超时场景测试通过")
    
    manager.stop()
    
    return True

def test_resource_allocation_callback():
    """测试资源分配回调"""
    print("\n" + "="*60)
    print("测试场景4: 资源分配回调函数")
    print("="*60)
    
    # 清理历史记录
    allocation_history.clear()
    
    # 创建管理器
    manager = DynamicScalingManager(
        worker_id="worker_1",
        total_workers=4,
        base_allocation=ResourceAllocation(
            max_concurrent_requests=10,
            rpm_limit=200,
            tpm_limit=10000,
            num_workers=10
        ),
        enable_scaling=True
    )
    
    manager.set_allocation_callback(test_callback)
    manager.start()
    
    # 标记为活跃
    manager.update_activity(is_active=True, current_load=0.9, active_tasks=8)
    
    # 等待调整
    time.sleep(12)
    
    # 检查回调是否被调用
    print(f"\n📞 回调调用次数: {len(allocation_history)}")
    assert len(allocation_history) > 0, "回调函数应该被调用"
    
    # 检查最后一次回调的分配
    last_allocation = allocation_history[-1]['allocation']
    print(f"📊 最后一次回调分配: {last_allocation}")
    
    assert last_allocation['max_concurrent_requests'] == 40, "应该分配全部并发数"
    assert last_allocation['rpm_limit'] == 800, "应该分配全部RPM"
    
    print("\n✅ 资源分配回调测试通过")
    
    manager.stop()
    
    return True

def main():
    """主测试函数"""
    print("\n" + "="*60)
    print("动态资源分配系统测试")
    print("="*60)
    
    tests = [
        ("单Worker场景", test_single_worker_scenario),
        ("多Worker场景", test_multiple_workers_scenario),
        ("Worker超时场景", test_worker_timeout_scenario),
        ("资源分配回调", test_resource_allocation_callback),
    ]
    
    passed = 0
    failed = 0
    
    for test_name, test_func in tests:
        try:
            if test_func():
                passed += 1
                print(f"\n✅ {test_name} 测试通过")
            else:
                failed += 1
                print(f"\n❌ {test_name} 测试失败")
        except Exception as e:
            failed += 1
            print(f"\n❌ {test_name} 测试异常: {e}")
            import traceback
            traceback.print_exc()
    
    # 清理测试文件
    state_file = Path.cwd() / ".worker_activity_state.json"
    if state_file.exists():
        state_file.unlink()
        print(f"\n🧹 已清理测试文件: {state_file}")
    
    # 输出总结
    print("\n" + "="*60)
    print("测试总结")
    print("="*60)
    print(f"✅ 通过: {passed}")
    print(f"❌ 失败: {failed}")
    print(f"📊 总计: {passed + failed}")
    
    if failed == 0:
        print("\n🎉 所有测试通过！")
        return 0
    else:
        print(f"\n⚠️  {failed} 个测试失败")
        return 1

if __name__ == "__main__":
    sys.exit(main())
