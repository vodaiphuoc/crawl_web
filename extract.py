import py_vncorenlp

# Automatically download VnCoreNLP components from the original repository
# and save them in some local working folder


# py_vncorenlp.download_model(save_dir=__file__.replace('extract.py','VnCoreNLP'))



# Load VnCoreNLP from the local working folder that contains both `VnCoreNLP-1.2.jar` and `models` 
model = py_vncorenlp.VnCoreNLP(annotators=["wseg", "pos"], save_dir=__file__.replace('extract.py','VnCoreNLP'))


# each segments are seperated by '.'
_wseg_re = model.word_segment(
"""Ngoài các cá nhân, một số tổ chức liên quan đến ông Bùi Thành Nhơn cũng sở hữu cổ phần tại Novaland. Cụ thể, 
Công ty cổ phần Novagroup, do ông Bùi Thành Nhơn giữ chức Chủ tịch HĐQT, nắm 343,8 triệu cổ phiếu, 
tỷ lệ 17,63% vốn. Một công ty khác cũng do ông Bùi Thành Nhơn giữ chức Chủ tịch HĐQT đó là CTCP Diamond 
Properties nắm 168,6 triệu cổ phiếu, tỷ lệ 8,65% vốn."""
)

for _seg in _wseg_re:

	# process segments
	components = [component.replace('_',' ') for component in _seg.split(' ') if component != ',']

	print(components)

	# _pos_re = model.annotate_text(_seg)
	# print(_seg)
	# print(_pos_re[0])

