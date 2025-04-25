import os
import google.generativeai as genai
import PIL.Image 
import argparse
import sys
import re # For parsing the result text

# --- API Key Configuration ---
# IMPORTANT: Use environment variables in production!
# api_key = os.environ.get("GEMINI_API_KEY") 
api_key = "AIzaSyD6d0iy27sTkwPT9YHhS4yoCaYzuuy5-Vo" # Replace with your key or use environment variable

if not api_key:
    print("오류: Gemini API 키가 필요합니다. 스크립트 내에 설정하거나 GEMINI_API_KEY 환경 변수를 사용하세요.")
    sys.exit(1)
elif api_key == "YOUR_API_KEY":
     print("오류: 스크립트 내의 'YOUR_API_KEY'를 실제 Gemini API 키로 변경해주세요.")
     sys.exit(1)

try:
    genai.configure(api_key=api_key)
    # Optionally check if configuration is successful, e.g., by listing models
    # genai.list_models() 
    print("Gemini API 설정 완료.")
except Exception as e:
    print(f"오류: Gemini API 설정 실패: {e}")
    sys.exit(1)
# --- End API Key Configuration ---


ALLOWED_EXTENSIONS = {'.png', '.jpg', '.jpeg', '.gif', '.webp'}
BATCH_SIZE = 5 # Process 5 images at a time

def parse_batch_result(batch_result_text, filenames_in_batch):
    """Parses the combined text result from a batch API call into individual results per filename."""
    results = []
    # Split the text based on the filename separator "--- 이미지: [filename] ---"
    # We need to handle the text before the first image marker as well (it might be general intro)
    
    # Create a regex pattern to capture filename and the text following it
    # Pattern looks for "--- 이미지: filename ---" and captures the text until the next marker or end of string
    pattern = r"--- 이미지: (.*?) ---\n(.*?)(?=--- 이미지: |$)"
    matches = re.findall(pattern, batch_result_text, re.DOTALL | re.IGNORECASE)
    
    parsed_results = {filename.strip(): text.strip() for filename, text in matches}

    # Match parsed results back to the original filenames in the batch order
    for filename in filenames_in_batch:
        if filename in parsed_results:
            result_text = parsed_results[filename]
            
            # 신뢰도 추출 (예: "신뢰도: 85%")
            confidence = "N/A"
            confidence_match = re.search(r'신뢰도:\s*(\d+)%', result_text)
            if confidence_match:
                confidence = confidence_match.group(1) + "%"
            
            results.append({'filename': filename, 'result_text': result_text, 'confidence': confidence})
        else:
            # Handle cases where a filename might not be found in the response
            # This could happen if the model failed to follow instructions for a specific image
            results.append({
                'filename': filename, 
                'result_text': '오류: 모델 응답에서 해당 파일의 결과를 찾을 수 없습니다.',
                'confidence': 'N/A'
            })
            
    # If parsing failed completely (no matches), return a generic error for all files
    if not results and filenames_in_batch:
        error_text = f"오류: 모델 응답 파싱 실패. 전체 응답:\n{batch_result_text}"
        results = [{'filename': fn, 'result_text': error_text, 'confidence': 'N/A'} for fn in filenames_in_batch]

    return results


def moderate_image_batch(image_paths):
    """Calls the Gemini API (GenerativeModel) to moderate a batch of images and returns parsed results."""
    print(f"--- 배치 검수 시작 ({len(image_paths)}개 이미지) ---")
    
    content = []
    valid_images_in_batch = [] # Store PIL Image objects for valid images
    filenames_in_batch = [] # Store filenames corresponding to valid_images_in_batch

    # Initial prompt for the batch - emphasizing filename separation
    prompt = """\
당신은 주어진 여러 이미지가 광고 소재로 사용 가능한지 판단하는 전문가입니다. 
아래 나열될 각 이미지에 대해 다음 기준에 따라 평가하고, 각 이미지별로 '사용 가능' 또는 '사용 불가' 판정을 내린 후 그 이유를 명확히 설명해주세요.
**각 이미지의 분석 결과 시작 부분에 반드시 "--- 이미지: [파일명] ---" 형식으로 파일명을 명시해주세요.**
**그리고 각 판단에 대한 신뢰도(Confidence) 레벨을 0~100% 사이의 숫자로 반드시 표시해 주세요. 예: "신뢰도: 85%"**

**이미지 사용 가능 여부 판단 기준:**

1.  **이미지 품질 및 심미성 (Quality & Aesthetics):**
    *   **가능:** 선명하고 초점이 잘 맞으며, 판매 상품의 형태가 명확하게 보이는 고해상도 이미지. 배경이 깔끔하거나 자연스럽게 아웃포커싱되어 제품/인물에 집중되는 이미지.
    *   **불가:** 저화질, 흐릿함, 심한 노이즈, 과도한 왜곡/보정, 복잡하거나 산만한 배경, 어둡거나 조명이 부적절한 이미지.

2.  **콘텐츠 적절성 (Content Appropriateness):**
    *   **불가:**
        *   혐오/차별/비방: 혐오감을 주거나 사회적 이슈/차별/비방 소지가 있는 콘텐츠.
        *   허위/과장: 실제 상품과 다르거나 기능을 과장하는 이미지 (특히 Before/After 비교 이미지).
        *   기타: 담배 노출 이미지, 과도한 분할컷/합성 이미지.
3. 선정성/과장성 판단 기준
    가능:
    속옷 상품에 대한 속옷 착용샷. 배경과 조화된 상태에서 모델이 착용한 사진
    불가능:
    특정 성적인 부위(엉덩이, 가슴골, Y존 등)가 클로즈업된 이미지. 모델이 착용했더라도 자세가 성적인 연상을 유발하는 이미지. 성인용품/이벤트성 속옷 등.
4.  **텍스트/로고/배지 (Text/Logos/Badges):**
    *   **상품형 광고:** 이미지 전체의 1/9 면적 이내 텍스트 허용 (상품명, 자체제작 로고, 감성적 문구 등). 가격/혜택 정보는 허용되나 과도한 홍보/후킹 문구, 두꺼운 서체로 이미지를 가리는 것 불가.
    *   **배너형 광고:** 이미지 내 텍스트, 일러스트, 특수효과 삽입 불가.
    *   **공통 불가:** 잘 알려지지 않은 브랜드 로고는 해당 상품의 로고일 가능성이 있으므로 무시, 유명 스포츠팀 로고/구단명, 브랜드 로고 무단 사용은 금지.

5.  **지적재산권 및 초상권 (IP & Portrait Rights):**
    *   **가능:** 직접 촬영/제작했거나 정당한 라이선스를 확보한 이미지.
    *   **불가:** 타인의 저작권(사진, 디자인), 상표권(로고, 브랜드명), 초상권(연예인, 유튜버 등 유명인)을 침해하는 이미지. (라이선스 증빙 불가한 상품 택 이미지 포함)
    *   **조건부 가능:** 캐릭터, 연예인/유명인 모델 사용, 방송 협찬 등은 사전 라이선스 제출 및 심사 완료 시 가능.

6.  **기술 사양 (Technical Specs):** (참고용)
    *   상품형: 권장 비율 1:1.2 (가로:세로), 권장 사이즈 600*720px 이상. GIF 첫 프레임 노출.
    *   배너형: 600*456px (PNG, JPG).
    *   공통 불가: 과도한 여백이 포함된 이미지.

--- 이제 아래 이미지들을 순서대로 분석해주세요 ---
"""
    content.append(prompt)

    # Load images as PIL objects and add to content list
    for filepath in image_paths:
        filename = os.path.basename(filepath)
        try:
            img = PIL.Image.open(filepath)
            content.append(f"--- 이미지: {filename} ---") # Add separator with filename
            content.append(img) # Add the PIL image object
            valid_images_in_batch.append(img)
            filenames_in_batch.append(filename) # Store filename for parsing later

        except FileNotFoundError:
            print(f"오류 (배치 내): 파일을 찾을 수 없습니다 - {filepath}")
        except PIL.UnidentifiedImageError:
             print(f"오류 (배치 내): 이미지 파일을 열 수 없습니다 - {filepath}")
        except Exception as e:
            print(f"오류 (배치 내): 이미지 로딩 중 오류 발생 ({filename}): {e}")

    if not valid_images_in_batch:
        print("--- 배치 검수 중단: 유효한 이미지가 없습니다. ---")
        # Return empty list or specific error structure if needed
        return [{'filename': fn, 'result_text': '오류: 이미지 로드 실패', 'confidence': 'N/A'} for fn in [os.path.basename(p) for p in image_paths]]

    # Call Gemini API using GenerativeModel
    batch_results = []
    try:
        # Using gemini-pro-vision as it's suited for image analysis
        model = genai.GenerativeModel('gemini-2.0-flash', generation_config={"temperature": 0.1}) 
        response = model.generate_content(content, stream=False)

        try:
            result_text = response.text
            print(f"--- 배치 원본 응답 수신 ({len(filenames_in_batch)}개 이미지)---") # Log reception
            # Parse the combined text response
            batch_results = parse_batch_result(result_text, filenames_in_batch)
            
        except ValueError:
             # Handle potential safety blocks 
             error_text = f"콘텐츠 안전 문제로 응답이 차단되었을 수 있습니다. Gemini 안전 설정을 확인하세요. Full Response: {response.prompt_feedback}"
             print(error_text)
             batch_results = [{'filename': fn, 'result_text': error_text, 'confidence': 'N/A'} for fn in filenames_in_batch]
        except Exception as e: # Catch parsing errors
            error_text = f"모델 응답 파싱 중 오류 발생: {e}\n원본 응답: {response.text if hasattr(response, 'text') else 'N/A'}"
            print(error_text)
            batch_results = [{'filename': fn, 'result_text': error_text, 'confidence': 'N/A'} for fn in filenames_in_batch]
        
        # Print individual parsed results to console
        print("--- 개별 결과 (콘솔) ---")
        for res in batch_results:
            print(f"파일: {res['filename']}")
            print(f"신뢰도: {res.get('confidence', 'N/A')}")
            print(f"결과:\n{res['result_text']}")
            print("---")
        print(f"--- 배치 검수 완료 ({len(batch_results)}/{len(image_paths)} 결과 처리) ---\n")


    except Exception as e:
        error_text = f"\nGemini API 호출 중 오류 발생 (파일: {filenames_in_batch}): {e}"
        print(error_text)
        print(f"--- 배치 검수 실패 ---\n")
        # Return error results for all images in the batch
        batch_results = [{'filename': fn, 'result_text': error_text, 'confidence': 'N/A'} for fn in filenames_in_batch]
        
    # Add full filepath back for HTML report linking
    final_results = []
    original_paths_dict = {os.path.basename(p): p for p in image_paths}
    for res in batch_results:
        res['filepath'] = original_paths_dict.get(res['filename'], res['filename']) # Fallback to filename if path not found
        final_results.append(res)

    return final_results


def generate_html_report(results, output_dir, image_source_dir_rel):
    """Generates an HTML report from the moderation results."""
    
    # Ensure image source dir is relative for HTML links
    image_source_dir_html = os.path.relpath(image_source_dir_rel, start=output_dir)

    html_content = """
<!DOCTYPE html>
<html lang="ko">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>이미지 검수 결과 보고서</title>
    <style>
        body { font-family: sans-serif; margin: 20px; }
        .result-item { 
            border: 1px solid #ccc; 
            margin-bottom: 20px; 
            padding: 15px; 
            display: flex; 
            flex-wrap: wrap; /* Allow wrapping on smaller screens */
            gap: 20px; /* Space between image and text */
            align-items: flex-start; /* Align items to the top */
        }
        .result-item img { 
            max-width: 200px; /* Limit image width */
            height: auto; 
            border: 1px solid #eee; 
        }
        .result-text { flex: 1; min-width: 300px; /* Ensure text area has minimum width */ }
        .result-text h3 { margin-top: 0; }
        .result-text pre { 
            white-space: pre-wrap; /* Wrap long lines */
            word-wrap: break-word; /* Break long words */
            background-color: #f8f8f8; 
            padding: 10px; 
            border: 1px solid #eee;
            font-size: 0.9em;
            max-height: 400px; /* Limit height and make scrollable */
            overflow-y: auto; /* Add scrollbar if needed */
        }
        .status-possible { color: green; font-weight: bold; }
        .status-impossible { color: red; font-weight: bold; }
        .status-error { color: orange; font-weight: bold; }
        .confidence-pill {
            display: inline-block;
            padding: 2px 8px;
            border-radius: 12px;
            background-color: #f0f0f0;
            font-size: 0.85em;
            margin-left: 10px;
        }
    </style>
</head>
<body>
    <h1>이미지 검수 결과 보고서</h1>
"""

    for item in results:
        filepath = item['filepath']
        filename = item['filename']
        result_text = item['result_text']
        confidence = item.get('confidence', 'N/A')
        
        # Determine status for styling (simple check)
        status_class = "status-error" # Default to error/unknown
        if "사용 가능" in result_text:
            status_class = "status-possible"
        elif "사용 불가" in result_text:
            status_class = "status-impossible"
        elif "오류:" in result_text: # Explicit error
             status_class = "status-error"

        # Construct relative image path for HTML
        # Assume image_source_dir_html is the relative path from HTML file to the image folder
        img_src = os.path.join(image_source_dir_html, filename)
        
        html_content += f"""
    <div class="result-item">
        <img src="{img_src}" alt="{filename}" title="{filepath}">
        <div class="result-text">
            <h3>{filename} (<span class="{status_class}">{status_class.split('-')[1].upper()}</span>) <span class="confidence-pill">신뢰도: {confidence}</span></h3>
            <pre>{result_text}</pre>
        </div>
    </div>
"""

    html_content += """
</body>
</html>
"""
    output_filepath = os.path.join(output_dir, "moderation_results.html")
    try:
        with open(output_filepath, 'w', encoding='utf-8') as f:
            f.write(html_content)
        print(f"\nHTML 보고서 생성 완료: {output_filepath}")
    except IOError as e:
        print(f"\n오류: HTML 보고서 파일 생성 실패 - {output_filepath}, {e}")


def main(directory):
    """Processes all images in the specified directory in batches and generates an HTML report."""
    if not os.path.isdir(directory):
        print(f"오류: 디렉토리를 찾을 수 없습니다 - {directory}")
        return

    print(f"\'{directory}\' 디렉토리에서 이미지 검수를 시작합니다 (배치 크기: {BATCH_SIZE}, 모델: gemini-pro-vision)...")
    
    image_files = []
    for filename in os.listdir(directory):
        # Check if filename itself contains invalid characters for path joining (less common)
        if any(c in filename for c in '\\/:*?\"<>|'): 
             print(f"경고: 파일명에 유효하지 않은 문자가 포함되어 건너<0xEB><0x9B><0x84>니다: {filename}")
             continue
             
        filepath = os.path.join(directory, filename)
        # Check if the combined path is valid before checking file type
        try:
            if os.path.isfile(filepath):
                _, ext = os.path.splitext(filename)
                if ext.lower() in ALLOWED_EXTENSIONS:
                    image_files.append(filepath)
        except OSError as e:
             print(f"경고: 파일 경로 접근 중 오류 발생하여 건너<0xEB><0x9B><0x84>니다: {filepath}, 오류: {e}")


    if not image_files:
        print(f"\'{directory}\' 디렉토리에서 검수할 이미지 파일을 찾지 못했습니다.")
        return

    all_results = []
    # Process images in batches
    for i in range(0, len(image_files), BATCH_SIZE):
        batch_paths = image_files[i:i + BATCH_SIZE]
        batch_results = moderate_image_batch(batch_paths)
        if batch_results: # Ensure we have results before extending
            all_results.extend(batch_results)
    
    print("\n모든 이미지 배치 처리 완료.")

    # Generate HTML report
    # Assume report is generated in the current working directory
    output_directory = "." 
    # Pass the original image directory path to calculate relative paths
    generate_html_report(all_results, output_directory, directory)


if __name__ == "__main__":
    # Setup Argument Parser
    parser = argparse.ArgumentParser(description='Gemini API(GenerativeModel)를 사용하여 이미지 배치 검수 및 HTML 보고서 생성')
    parser.add_argument(
        'directory',
        nargs='?',
        default='./image',
        help='검수할 이미지 파일들이 있는 디렉토리 경로 (기본값: ./image)'
    )
    # API Key argument is removed, using global configuration at the top
    args = parser.parse_args()

    # API Key configuration happens at the top now

    main(args.directory) 