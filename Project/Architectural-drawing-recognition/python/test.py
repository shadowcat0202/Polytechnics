import pprint

import PyPDF2   # 파이썬에서 PDF 페이지 추출, 병합, 핸들링
from PIL import Image   # 파이썬에서 이미지 처리 패키지

file_name = "./test_pdf.pdf"
pdf_file = PyPDF2.PdfFileReader(open(file_name, "rb"))
for i in range(pdf_file.getNumPages()):
    page = pdf_file.getPage(i)
    print(f"{i + 1} 페이지===============================================")
    if '/XObject' in page["/Resources"]:
        xObject = page["/Resources"]["/XObject"].getObject()
        for obj in xObject:
            print(f"==-==-==-==-==-==-==")
            if xObject[obj]['/Subtype'] == "/Image":
                pprint.pprint(xObject[obj]['/Subtype'])
                size = (xObject[obj]["/Width"], xObject[obj]["/Height"])
                data = xObject[obj].getData()
                # pprint.pprint(xObject[obj])
                if "/ColorSpace" in xObject[obj]:
                    if xObject[obj]['/ColorSpace'] == '/DeviceRGB':
                        mode = "RGB"
                    else:
                        mode = "P"
                else:
                    mode = "RGB"
                pprint.pprint(xObject[obj])
                if xObject[obj]['/Filter'] == '/FlateDecode':
                    img = Image.frombytes(mode, size, data)
                    """
                    mode : 
                    1 (1-bit pixels, black and white, stored with one pixel per byte)
                    L (8-bit pixels, black and white)
                    P (8-bit pixels, mapped to any other mode using a color palette)
                    RGB (3x8-bit pixels, true color)
                    RGBA (4x8-bit pixels, true color with transparency mask)
                    CMYK (4x8-bit pixels, color separation)
                    YCbCr (3x8-bit pixels, color video format)
                    Note that this refers to the JPEG, and not the ITU-R BT.2020, standard
                    LAB (3x8-bit pixels, the L*a*b color space)
                    HSV (3x8-bit pixels, Hue, Saturation, Value color space)
                    I (32-bit signed integer pixels)
                    F (32-bit floating point pixels)
                    """
                    img.save(obj[1:] + ".png")
                elif xObject[obj]['/Filter'] == '/DCTDecode':
                    img = open(obj[1:] + ".jpg", "wb")
                    img.write(data)
                    img.close()
                elif xObject[obj]['/Filter'] == '/JPXDecode':
                    img = open(obj[1:] + ".jp2", "wb")
                    img.write(data)
                    img.close()
    else:
        pass

