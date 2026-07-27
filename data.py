# ============================================================================
# data.py -- PEMBACA train.CSV / test.CSV hasil GetData.py.
#
# Dipanggil sekali di RunModel.main() untuk train dan sekali untuk test.
# Mengubah file CSV datar menjadi list tuple, satu tuple per siswa.
# ============================================================================
import csv
import random


def load_data(fileName):
    # Return:
    #   tuple_rows        : list tuple siswa, tiap tuple =
    #                       ([n], [indeks soal], [correctness], baris attempt x n)
    #   max_num_problems  : jumlah attempt TERBANYAK -> dipakai sebagai T
    #   max_skill_num + 1 : jumlah soal unik (indeks 0-based -> +1)
    rows = []
    max_skill_num = 0
    max_num_problems = 0
    with open(fileName, "r", encoding='utf-8-sig') as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        for row in reader:
            rows.append(row)
    # Baca seluruh file ke memori dulu (dataset kecil, aman).
    index = 0
    print("the number of rows is " + str(len(rows)))
    tuple_rows = []
    # turn list to tuple
    while (index < len(rows) - 1):
        # Struktur satu blok siswa: 3 baris header + problems_num baris attempt.
        # rows[index+1] = daftar indeks soal -> max-nya dipakai menghitung
        # jumlah skill/soal unik.
        problems_num = int(rows[index][0])
        tmp_max_skill = max(map(int, rows[index + 1]))
        # Siswa dengan <= 1 attempt DIBUANG: tidak ada pasangan
        # (attempt sekarang -> attempt berikutnya) untuk dijadikan label.
        if (problems_num <= 1):
            index = index + 3 + problems_num
        else:
            # Ambil satu blok utuh jadi tuple, lalu lompat ke siswa berikutnya.
            if problems_num > max_num_problems:
                max_num_problems = problems_num
            tup = (rows[index + i] for i in range(int(rows[index][0]) + 3))
            tup = tuple(tup)
            '''
            tup = (rows[index], rows[index+1], rows[index+2],rows[index+3])
            '''
            tuple_rows.append(tup)
            index += int(rows[index][0]) + 3

        if (tmp_max_skill > max_skill_num):
            max_skill_num = tmp_max_skill
    # shuffle the tuple
    # Acak urutan siswa supaya batch tidak berisi siswa yang berdekatan/mirip.
    # Hasilnya tetap deterministik karena RunModel.py memanggil random.seed(1).
    # CATATAN: ini juga mengacak test set -- gunakan pembacaan manual (lihat
    # inference.LoadTestStudents) kalau butuh urutan tetap.
    random.shuffle(tuple_rows)
    print("The number of students is ", len(tuple_rows))
    print("Finish reading data")
    print('max_num_probles:', max_num_problems,
          'max_skill_num:', max_skill_num + 1)
    return tuple_rows, max_num_problems, max_skill_num + 1
