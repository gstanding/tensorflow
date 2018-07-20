inputs: [m, 299, 299, 3]

'InceptionV3'

conv2d 32 [3, 3] stride=2 [149, 149, 32] 

conv2d 32 [3, 3] stride=1 [147, 147, 32]

conv2d 64 [3, 3] padding='SAME' [147, 147, 64]

max_pool2d [3, 3] stride=2 [73, 73, 64] 

conv2d 80 [1, 1] stride=1 [71, 71, 80]

conv2d 192 [3, 3] stride=1 [71, 71, 192]

max_pool2d [3, 3] stride=2 [35, 35, 192]

padding='SAME'

'InceptionV3/Mixed_5b'

'Branch_0' 

conv2d 64 [1, 1] stride=1 [35, 35, 64]

'Branch_1'

conv2d 48 [1, 1] stride=1 [35, 35, 48]

conv2d 64 [5, 5] stride=1 [35, 35, 64]

'Branch_2'

conv2d 64 [1, 1] stride=1 [35, 35, 64]

conv2d 96 [3, 3] stride=1 [35, 35, 96]

conv2d 96 [3, 3] stride=1 [35, 35, 96]

'Branch_3'

avg_pool2d [3, 3] stride=1 [35, 35, 192]

conv2d 32 [3, 3] stride=1 [35, 35, 32]

net = tf.concat([branch_0, branch_1, branch_2, branch_3], 3) [35, 35, 256]

'InceptionV3/Mixed_5c'

'Branch_0' 

conv2d 64 [1, 1] stride=1 [35, 35, 64]

'Branch_1'

conv2d 48 [1, 1] stride=1 [35, 35, 48]

conv2d 64 [5, 5] stride=1 [35, 35, 64]

'Branch_2'

conv2d 64 [1, 1] stride=1 [35, 35, 64]

conv2d 96 [3, 3] stride=1 [35, 35, 96]

conv2d 96 [3, 3] stride=1 [35, 35, 96]

'Branch_3'

avg_pool2d [3, 3] stride=1 [35, 35, 192]

conv2d 64 [3, 3] stride=1 [35, 35, 64]

net = tf.concat([branch_0, branch_1, branch_2, branch_3], 3) [35, 35, 288]

'InceptionV3/Mixed_5d'

'Branch_0' 

conv2d 64 [1, 1] stride=1 [35, 35, 64]

'Branch_1'

conv2d 48 [1, 1] stride=1 [35, 35, 48]

conv2d 64 [5, 5] stride=1 [35, 35, 64]

'Branch_2'

conv2d 64 [1, 1] stride=1 [35, 35, 64]

conv2d 96 [3, 3] stride=1 [35, 35, 96]

conv2d 96 [3, 3] stride=1 [35, 35, 96]

'Branch_3'

avg_pool2d [3, 3] stride=1 [35, 35, 192]

conv2d 64 [3, 3] stride=1 [35, 35, 64]

net = tf.concat([branch_0, branch_1, branch_2, branch_3], 3) [35, 35, 288]

'InceptionV3/Mixed_6a'

'Branch_0'

conv2d 384 [3, 3] stride=2 padding='VALID' [17, 17, 384]

'Branch_1'

conv2d 64 [1, 1] stride=1 [35, 35, 64]

conv2d 96 [3, 3] stride=1 [35, 35, 96]

conv2d 96 [3, 3] stride=2 [17, 17, 96]

'Branch_2'

max_pool2d [3, 3] stride=2 [17, 17, 288]

net = tf.concat([branch_0, branch_1, branch_2], 3) [17, 17, 768]

'InceptionV3/Mixed_6b'

'Branch_0'

