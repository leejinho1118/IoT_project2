"""데이터 준비: Data/Prepared_dataset.z01 + .zip(분할 압축)을 이어 붙여 Data/N-BaIoT/ 로 풀어준다.
실행(저장소 FedMSE/ 에서): python reproduce/00_prepare_data.py
결과: Data/N-BaIoT/{IID-10-Client_Data, IID-50-Client_Data, nonIID-50-Client_Data, noniid-10-Client_Data}
(표준 라이브러리만 사용. 이미 풀려 있으면 아무것도 하지 않는다.)"""
import io, os, sys, shutil, struct, zipfile, zlib
D = "Data"; OUT = os.path.join(D, "N-BaIoT")
if os.path.isdir(os.path.join(OUT, "noniid-10-Client_Data")):
    print("이미 준비됨:", OUT); sys.exit(0)
part1 = open(os.path.join(D, "Prepared_dataset.z01"), "rb").read()
data = part1 + open(os.path.join(D, "Prepared_dataset.zip"), "rb").read()
# 분할 zip은 항목 오프셋이 조각 기준이라 파이썬 zipfile이 바로 못 읽는다. 중앙 디렉터리(이름·압축크기)만 쓰고,
# 데이터는 이어 붙인 바이트열에서 로컬 헤더를 순서대로 읽어 직접 푼다.
info = {zi.filename: zi for zi in zipfile.ZipFile(io.BytesIO(data)).infolist()}
pos = 4 if data[:4] == b"PK\x07\x08" else 0      # 분할 표식
tmp = os.path.join(D, "_extract_tmp"); n = 0
while data[pos:pos + 4] == b"PK\x03\x04":
    nl, el = struct.unpack("<HH", data[pos + 26:pos + 30])
    name = data[pos + 30:pos + 30 + nl].decode("utf-8", "replace"); zi = info[name]
    start = pos + 30 + nl + el; raw = data[start:start + zi.compress_size]; pos = start + zi.compress_size
    if name.endswith("/"): os.makedirs(os.path.join(tmp, name), exist_ok=True); continue
    body = raw if zi.compress_type == 0 else zlib.decompress(raw, -15)
    assert len(body) == zi.file_size and zlib.crc32(body) == zi.CRC, name
    out = os.path.join(tmp, name); os.makedirs(os.path.dirname(out), exist_ok=True); open(out, "wb").write(body); n += 1
src = os.path.join(tmp, "Scenarios_dataset", "Scenarios_dataset")
shutil.move(src, OUT); shutil.rmtree(tmp)
print(f"완료: 파일 {n}개 (CRC 검증 통과)", sorted(os.listdir(OUT)))
