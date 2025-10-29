#!/usr/bin/env python3
"""
GaitDynamics Pre-trained Model 간단 테스트
"""

import os
import torch
import sys

def test_basic_imports():
    """기본 import 테스트"""
    print("🔄 기본 패키지 import 테스트...")
    
    try:
        import numpy as np
        print("✅ NumPy")
        
        import pandas as pd  
        print("✅ Pandas")
        
        import torch
        print(f"✅ PyTorch {torch.__version__}")
        
        import nimblephysics as nimble
        print("✅ NimblePhysics")
        
        return True
    except Exception as e:
        print(f"❌ Import 실패: {e}")
        return False

def test_cuda():
    """CUDA 환경 테스트"""
    print("\n🔄 CUDA 환경 테스트...")
    
    print(f"CUDA 사용 가능: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU 개수: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
        
        # GPU 메모리 확인
        for i in range(torch.cuda.device_count()):
            memory_total = torch.cuda.get_device_properties(i).total_memory / 1024**3
            print(f"  GPU {i} 메모리: {memory_total:.1f} GB")
    
    return torch.cuda.is_available()

def test_model_files():
    """Pre-trained 모델 파일 존재 확인"""
    print("\n🔄 Pre-trained 모델 파일 확인...")
    
    base_path = "/home/ghlee/GaitDynamics/example_usage"
    
    diffusion_model = os.path.join(base_path, "GaitDynamicsDiffusion.pt")
    refinement_model = os.path.join(base_path, "GaitDynamicsRefinement.pt")
    osim_model = os.path.join(base_path, "example_opensim_model.osim")
    
    files_to_check = {
        "Diffusion Model": diffusion_model,
        "Refinement Model": refinement_model, 
        "OpenSim Model": osim_model
    }
    
    all_exist = True
    for name, path in files_to_check.items():
        if os.path.exists(path):
            size_mb = os.path.getsize(path) / 1024 / 1024
            print(f"✅ {name}: {size_mb:.1f} MB")
        else:
            print(f"❌ {name}: 파일 없음")
            all_exist = False
    
    return all_exist

def test_model_loading():
    """실제 모델 로딩 테스트"""
    print("\n🔄 모델 로딩 테스트...")
    
    try:
        # 간단한 체크포인트 로딩
        diffusion_path = "/home/ghlee/GaitDynamics/example_usage/GaitDynamicsDiffusion.pt"
        refinement_path = "/home/ghlee/GaitDynamics/example_usage/GaitDynamicsRefinement.pt"
        
        print("📁 Diffusion 모델 로딩 중...")
        diffusion_checkpoint = torch.load(diffusion_path, map_location='cpu', weights_only=False)
        print(f"   키들: {list(diffusion_checkpoint.keys())}")
        
        print("📁 Refinement 모델 로딩 중...")  
        refinement_checkpoint = torch.load(refinement_path, map_location='cpu', weights_only=False)
        print(f"   키들: {list(refinement_checkpoint.keys())}")
        
        print("✅ 모델 파일 로딩 성공!")
        return True
        
    except Exception as e:
        print(f"❌ 모델 로딩 실패: {e}")
        return False

def main():
    print("=" * 60)
    print("GaitDynamics Pre-trained Model 환경 테스트")
    print("=" * 60)
    
    # 기본 import 테스트
    if not test_basic_imports():
        print("\n❌ 기본 패키지 import 실패. 환경을 확인하세요.")
        return
    
    # CUDA 테스트  
    cuda_available = test_cuda()
    
    # 모델 파일 확인
    if not test_model_files():
        print("\n❌ 필요한 모델 파일들이 없습니다.")
        return
    
    # 모델 로딩 테스트
    if not test_model_loading():
        print("\n❌ 모델 로딩에 실패했습니다.")
        return
    
    print("\n" + "=" * 60)
    print("🎉 모든 기본 테스트 통과!")
    print("=" * 60)
    print("\n📋 환경 요약:")
    print(f"   - PyTorch: {torch.__version__}")
    print(f"   - CUDA: {'사용 가능' if cuda_available else '사용 불가'}")
    print(f"   - Pre-trained 모델: 정상")
    print("\n🚀 이제 실제 추론을 실행할 수 있습니다!")
    
    print("\n📝 다음 단계:")
    print("1. .mot 파일 (운동학적 데이터) 준비")
    print("2. .osim 파일 (인체 모델) 준비")  
    print("3. 추론 스크립트 실행")

if __name__ == "__main__":
    main()