import os
from test import predict, compare_rating
from process_voice import readfile, write_file_mic, show_mfcc

Choice = input(
    "Các chức năng của chương trình"
    "\nNhập 1 để nhận dạng"
    "\nNhập 2 để xem thông tin âm thanh"
    "\nNhập 3 để đánh giá nhận dạng"
    "\nNhâp 4 để nhập âm thanh trực tiếp từ mic"
    "\n \t Chọn chức năng muốn thực hiện? ")
if Choice == '1':
    predict()
elif Choice == '2':
    directory = os.getcwd() + '/train/'
    file_name = input("Nhập tên âm thanh cần trích xuất: ")
    mffc_fe = readfile(directory, file_name)
    show_mfcc(mffc_fe)
elif Choice == '3':
    compare_rating()
elif Choice == '4':
    write_file_mic()
