#!/usr/bin/env python3

import h5py
import numpy as np
import matplotlib.pyplot as plt
import os
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

def explore_h5_structure(file_path):
    """H5 파일의 전체 구조를 탐색하고 텍스트 파일로 저장합니다."""
    # 출력 파일명 생성 (현재 스크립트 폴더에 저장)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    base_name = os.path.splitext(os.path.basename(file_path))[0]
    output_file = os.path.join(script_dir, f"{base_name}_explored.txt")
    
    print(f"\n🔍 H5 파일 구조 분석 중: {file_path}")
    print(f"📝 결과를 {output_file}에 저장합니다...")
    
    try:
        with h5py.File(file_path, 'r') as f:
            with open(output_file, 'w', encoding='utf-8') as out:
                # 헤더 정보
                out.write(f"🔍 H5 파일 구조 분석 리포트\n")
                out.write(f"{'=' * 80}\n")
                out.write(f"📁 파일 경로: {file_path}\n")
                out.write(f"📁 파일 크기: {os.path.getsize(file_path) / (1024**2):.2f} MB\n")
                out.write(f"🕒 분석 시간: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                out.write(f"🔑 최상위 키 개수: {len(f.keys())}\n")
                out.write(f"🗂️ 최상위 키 목록: {list(f.keys())}\n\n")
                out.write(f"{'=' * 80}\n")
                out.write(f"📊 전체 구조 (상세)\n")
                out.write(f"{'=' * 80}\n\n")
                
                # 재귀적으로 모든 구조 탐색
                def write_structure(name, obj, level=0):
                    indent = "  " * level
                    if isinstance(obj, h5py.Group):
                        out.write(f"{indent}📂 그룹: {name}\n")
                        out.write(f"{indent}   └─ 하위 항목 수: {len(obj.keys())}\n")
                        if len(obj.keys()) > 0:
                            out.write(f"{indent}   └─ 하위 키: {list(obj.keys())}\n")
                    elif isinstance(obj, h5py.Dataset):
                        out.write(f"{indent}📊 데이터셋: {name}\n")
                        out.write(f"{indent}   ├─ 형태: {obj.shape}\n")
                        out.write(f"{indent}   ├─ 데이터 타입: {obj.dtype}\n")
                        out.write(f"{indent}   ├─ 크기: {obj.size:,} 요소\n")
                        
                        # 속성 정보
                        if len(obj.attrs) > 0:
                            out.write(f"{indent}   ├─ 속성:\n")
                            for attr_name, attr_value in obj.attrs.items():
                                out.write(f"{indent}   │  └─ {attr_name}: {attr_value}\n")
                        
                        # 데이터 미리보기 (소량)
                        if obj.size > 0 and len(obj.shape) <= 2:
                            preview_size = min(5, obj.shape[0] if len(obj.shape) > 0 else 1)
                            try:
                                if len(obj.shape) == 1:
                                    preview = obj[:preview_size]
                                elif len(obj.shape) == 2:
                                    preview = obj[:preview_size, :min(5, obj.shape[1])]
                                else:
                                    preview = "복잡한 다차원 데이터"
                                out.write(f"{indent}   └─ 미리보기: {preview}\n")
                            except Exception as e:
                                out.write(f"{indent}   └─ 미리보기 실패: {e}\n")
                        out.write("\n")
                
                # 전체 구조 출력
                f.visititems(write_structure)
                
                # 요약 정보 추가
                out.write(f"\n{'=' * 80}\n")
                out.write(f"📋 요약 정보\n")
                out.write(f"{'=' * 80}\n")
                
                # 데이터셋 개수 세기
                dataset_count = 0
                group_count = 0
                
                def count_items(name, obj):
                    nonlocal dataset_count, group_count
                    if isinstance(obj, h5py.Dataset):
                        dataset_count += 1
                    elif isinstance(obj, h5py.Group):
                        group_count += 1
                
                f.visititems(count_items)
                
                out.write(f"📊 총 데이터셋 수: {dataset_count}개\n")
                out.write(f"📂 총 그룹 수: {group_count}개\n")
                out.write(f"🗂️ 최상위 키: {list(f.keys())}\n")
                
                # 시계열 데이터 후보 찾기
                out.write(f"\n🕒 시계열 분석 가능한 데이터셋:\n")
                out.write(f"{'-' * 50}\n")
                
                def find_timeseries(name, obj):
                    if isinstance(obj, h5py.Dataset) and len(obj.shape) >= 1:
                        if obj.shape[0] > 10:  # 충분한 데이터 포인트
                            out.write(f"  🔸 {name}: {obj.shape} ({obj.dtype})\n")
                
                f.visititems(find_timeseries)
            
            print(f"✅ 구조 분석 완료! 결과 저장: {output_file}")
            return list(f.keys())
            
    except Exception as e:
        print(f"❌ 파일 읽기 오류: {e}")
        return []

def plot_timeseries(file_path, dataset_path):
    """시계열 데이터를 플롯합니다."""
    try:
        with h5py.File(file_path, 'r') as f:
            if dataset_path not in f:
                print(f"❌ 데이터셋을 찾을 수 없습니다: {dataset_path}")
                return
            
            dataset = f[dataset_path]
            
            if not isinstance(dataset, h5py.Dataset):
                print(f"❌ {dataset_path}는 데이터셋이 아닙니다.")
                return
            
            data = dataset[:]
            
            if len(data.shape) == 1:
                # 1차원 데이터
                plt.figure(figsize=(12, 6))
                plt.plot(data)
                plt.title(f"시계열 데이터: {dataset_path}")
                plt.xlabel("인덱스")
                plt.ylabel("값")
                plt.grid(True)
                plt.show()
                
            elif len(data.shape) == 2:
                # 2차원 데이터 - 첫 번째 몇 개 컬럼만 플롯
                plt.figure(figsize=(12, 8))
                max_cols = min(5, data.shape[1])
                
                for i in range(max_cols):
                    plt.subplot(max_cols, 1, i+1)
                    plt.plot(data[:, i])
                    plt.title(f"{dataset_path} - 컬럼 {i}")
                    plt.grid(True)
                
                plt.tight_layout()
                plt.show()
                
            else:
                print(f"❌ 플롯할 수 없는 데이터 형태: {data.shape}")
                
    except Exception as e:
        print(f"❌ 플롯 생성 오류: {e}")

def main():
    """메인 실행 함수"""
    # 기본 파일 경로
    default_file = "/home/exolabshare/datasets/combined_data.h5"
    
    print("🔍 H5 파일 탐색기")
    print("=" * 50)
    
    # 파일 경로 입력받기
    file_path = input(f"📁 H5 파일 경로를 입력하세요 (기본값: {default_file}): ").strip()
    if not file_path:
        file_path = default_file
    
    # 파일 존재 확인
    if not os.path.exists(file_path):
        print(f"❌ 파일을 찾을 수 없습니다: {file_path}")
        return
    
    # 1. 구조 탐색 및 텍스트 파일 저장
    keys = explore_h5_structure(file_path)
    
    if not keys:
        print("❌ 파일 구조를 읽을 수 없습니다.")
        return
    
    # 2. 시각화 옵션
    print(f"\n📊 시각화 옵션:")
    print(f"   1️⃣ 특정 데이터셋 시계열 플롯")
    print(f"   2️⃣ 종료")
    
    while True:
        choice = input("\n선택하세요 (1-2): ").strip()
        
        if choice == "1":
            # 데이터셋 경로 입력받기
            dataset_path = input("📊 플롯할 데이터셋 경로를 입력하세요 (예: /group/dataset): ").strip()
            if dataset_path:
                plot_timeseries(file_path, dataset_path)
            else:
                print("❌ 유효한 데이터셋 경로를 입력하세요.")
                
        elif choice == "2":
            print("👋 종료합니다.")
            break
        else:
            print("❌ 잘못된 선택입니다. 1 또는 2를 입력하세요.")

if __name__ == "__main__":
    main()
