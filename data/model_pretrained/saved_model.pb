нк 
Д╘
D
AddV2
x"T
y"T
z"T"
Ttype:
2	АР
B
AssignVariableOp
resource
value"dtype"
dtypetypeИ
~
BiasAdd

value"T	
bias"T
output"T" 
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
8
Const
output"dtype"
valuetensor"
dtypetype
.
Identity

input"T
output"T"	
Ttype
\
	LeakyRelu
features"T
activations"T"
alphafloat%═╠L>"
Ttype0:
2
,
Log
x"T
y"T"
Ttype:

2
q
MatMul
a"T
b"T
product"T"
transpose_abool( "
transpose_bbool( "
Ttype:

2	
Н
Mean

input"T
reduction_indices"Tidx
output"T"
	keep_dimsbool( " 
Ttype:
2	"
Tidxtype0:
2	
e
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool(И
?
Mul
x"T
y"T
z"T"
Ttype:
2	Р

NoOp
M
Pack
values"T*N
output"T"
Nint(0"	
Ttype"
axisint 
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
@
ReadVariableOp
resource
value"dtype"
dtypetypeИ
@
RealDiv
x"T
y"T
z"T"
Ttype:
2	
E
Relu
features"T
activations"T"
Ttype:
2	
o
	RestoreV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0И
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0И
?
Select
	condition

t"T
e"T
output"T"	
Ttype
H
ShardedFilename
basename	
shard

num_shards
filename
9
Softmax
logits"T
softmax"T"
Ttype:
2
-
Sqrt
x"T
y"T"
Ttype:

2
3
Square
x"T
y"T"
Ttype:
2
	
╛
StatefulPartitionedCall
args2Tin
output2Tout"
Tin
list(type)("
Tout
list(type)("	
ffunc"
configstring "
config_protostring "
executor_typestring И
@
StaticRegexFullMatch	
input

output
"
patternstring
N

StringJoin
inputs*N

output"
Nint(0"
	separatorstring 
<
Sub
x"T
y"T
z"T"
Ttype:
2	
М
Sum

input"T
reduction_indices"Tidx
output"T"
	keep_dimsbool( " 
Ttype:
2	"
Tidxtype0:
2	
Ц
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 И"serve*2.5.02v2.5.0-rc3-213-ga4dfb8d1a718Н■
Е
encoder2/fcn1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	Аx*%
shared_nameencoder2/fcn1/kernel
~
(encoder2/fcn1/kernel/Read/ReadVariableOpReadVariableOpencoder2/fcn1/kernel*
_output_shapes
:	Аx*
dtype0
|
encoder2/fcn1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:x*#
shared_nameencoder2/fcn1/bias
u
&encoder2/fcn1/bias/Read/ReadVariableOpReadVariableOpencoder2/fcn1/bias*
_output_shapes
:x*
dtype0
Е
encoder1/fcn1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	Аd*%
shared_nameencoder1/fcn1/kernel
~
(encoder1/fcn1/kernel/Read/ReadVariableOpReadVariableOpencoder1/fcn1/kernel*
_output_shapes
:	Аd*
dtype0
|
encoder1/fcn1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:d*#
shared_nameencoder1/fcn1/bias
u
&encoder1/fcn1/bias/Read/ReadVariableOpReadVariableOpencoder1/fcn1/bias*
_output_shapes
:d*
dtype0
П
encoder1/fcn3_gene/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	Аd**
shared_nameencoder1/fcn3_gene/kernel
И
-encoder1/fcn3_gene/kernel/Read/ReadVariableOpReadVariableOpencoder1/fcn3_gene/kernel*
_output_shapes
:	Аd*
dtype0
Ж
encoder1/fcn3_gene/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:d*(
shared_nameencoder1/fcn3_gene/bias

+encoder1/fcn3_gene/bias/Read/ReadVariableOpReadVariableOpencoder1/fcn3_gene/bias*
_output_shapes
:d*
dtype0
О
encoder2/fcn2/mean/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:x`**
shared_nameencoder2/fcn2/mean/kernel
З
-encoder2/fcn2/mean/kernel/Read/ReadVariableOpReadVariableOpencoder2/fcn2/mean/kernel*
_output_shapes

:x`*
dtype0
Ж
encoder2/fcn2/mean/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:`*(
shared_nameencoder2/fcn2/mean/bias

+encoder2/fcn2/mean/bias/Read/ReadVariableOpReadVariableOpencoder2/fcn2/mean/bias*
_output_shapes
:`*
dtype0
О
encoder1/fcn2/mean/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:dd**
shared_nameencoder1/fcn2/mean/kernel
З
-encoder1/fcn2/mean/kernel/Read/ReadVariableOpReadVariableOpencoder1/fcn2/mean/kernel*
_output_shapes

:dd*
dtype0
Ж
encoder1/fcn2/mean/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:d*(
shared_nameencoder1/fcn2/mean/bias

+encoder1/fcn2/mean/bias/Read/ReadVariableOpReadVariableOpencoder1/fcn2/mean/bias*
_output_shapes
:d*
dtype0
Р
decoder1/layer_gene/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:dd*+
shared_namedecoder1/layer_gene/kernel
Й
.decoder1/layer_gene/kernel/Read/ReadVariableOpReadVariableOpdecoder1/layer_gene/kernel*
_output_shapes

:dd*
dtype0
И
decoder1/layer_gene/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:d*)
shared_namedecoder1/layer_gene/bias
Б
,decoder1/layer_gene/bias/Read/ReadVariableOpReadVariableOpdecoder1/layer_gene/bias*
_output_shapes
:d*
dtype0
Е
decoder1/fcn1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	dА	*%
shared_namedecoder1/fcn1/kernel
~
(decoder1/fcn1/kernel/Read/ReadVariableOpReadVariableOpdecoder1/fcn1/kernel*
_output_shapes
:	dА	*
dtype0
}
decoder1/fcn1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:А	*#
shared_namedecoder1/fcn1/bias
v
&decoder1/fcn1/bias/Read/ReadVariableOpReadVariableOpdecoder1/fcn1/bias*
_output_shapes	
:А	*
dtype0
Е
decoder2/fcn1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	`А	*%
shared_namedecoder2/fcn1/kernel
~
(decoder2/fcn1/kernel/Read/ReadVariableOpReadVariableOpdecoder2/fcn1/kernel*
_output_shapes
:	`А	*
dtype0
}
decoder2/fcn1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:А	*#
shared_namedecoder2/fcn1/bias
v
&decoder2/fcn1/bias/Read/ReadVariableOpReadVariableOpdecoder2/fcn1/bias*
_output_shapes	
:А	*
dtype0
Ж
decoder1/fcn2/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
А	А*%
shared_namedecoder1/fcn2/kernel

(decoder1/fcn2/kernel/Read/ReadVariableOpReadVariableOpdecoder1/fcn2/kernel* 
_output_shapes
:
А	А*
dtype0
}
decoder1/fcn2/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*#
shared_namedecoder1/fcn2/bias
v
&decoder1/fcn2/bias/Read/ReadVariableOpReadVariableOpdecoder1/fcn2/bias*
_output_shapes	
:А*
dtype0
Ж
decoder2/fcn2/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
А	А*%
shared_namedecoder2/fcn2/kernel

(decoder2/fcn2/kernel/Read/ReadVariableOpReadVariableOpdecoder2/fcn2/kernel* 
_output_shapes
:
А	А*
dtype0
}
decoder2/fcn2/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*#
shared_namedecoder2/fcn2/bias
v
&decoder2/fcn2/bias/Read/ReadVariableOpReadVariableOpdecoder2/fcn2/bias*
_output_shapes	
:А*
dtype0
Д
encoder1/fcn3/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:dP*%
shared_nameencoder1/fcn3/kernel
}
(encoder1/fcn3/kernel/Read/ReadVariableOpReadVariableOpencoder1/fcn3/kernel*
_output_shapes

:dP*
dtype0
|
encoder1/fcn3/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:P*#
shared_nameencoder1/fcn3/bias
u
&encoder1/fcn3/bias/Read/ReadVariableOpReadVariableOpencoder1/fcn3/bias*
_output_shapes
:P*
dtype0
Д
encoder2/fcn3/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:`P*%
shared_nameencoder2/fcn3/kernel
}
(encoder2/fcn3/kernel/Read/ReadVariableOpReadVariableOpencoder2/fcn3/kernel*
_output_shapes

:`P*
dtype0
|
encoder2/fcn3/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:P*#
shared_nameencoder2/fcn3/bias
u
&encoder2/fcn3/bias/Read/ReadVariableOpReadVariableOpencoder2/fcn3/bias*
_output_shapes
:P*
dtype0
J
ConstConst*
_output_shapes
: *
dtype0*
valueB
 *═╠L>

NoOpNoOp
 S
Const_1Const"/device:CPU:0*
_output_shapes
: *
dtype0*╕S
valueоSBлS BдS
╞
layer-0
layer-1
layer-2
layer_with_weights-0
layer-3
layer-4
layer-5
layer_with_weights-1
layer-6
layer_with_weights-2
layer-7
	layer_with_weights-3
	layer-8

layer-9
layer-10
layer-11
layer-12
layer-13
layer-14
layer_with_weights-4
layer-15
layer-16
layer-17
layer-18
layer-19
layer-20
layer_with_weights-5
layer-21
layer-22
layer-23
layer-24
layer-25
layer-26
layer-27
layer-28
layer-29
layer-30
 layer-31
!layer_with_weights-6
!layer-32
"layer_with_weights-7
"layer-33
#layer-34
$layer-35
%layer_with_weights-8
%layer-36
&layer_with_weights-9
&layer-37
'layer-38
(layer-39
)layer-40
*layer-41
+layer-42
,layer-43
-layer-44
.layer-45
/layer-46
0layer-47
1layer-48
2layer-49
3layer-50
4layer-51
5layer-52
6layer-53
7layer_with_weights-10
7layer-54
8layer_with_weights-11
8layer-55
9layer-56
:layer-57
;layer-58
<layer-59
=layer-60
>layer-61
?layer-62
@layer-63
Alayer-64
Blayer-65
Clayer-66
Dlayer-67
Elayer-68
Flayer-69
Glayer-70
Hlayer-71
Ilayer-72
Jlayer-73
Klayer-74
Llayer-75
Mlayer-76
Nlayer-77
Olayer-78
Player-79
Qlayer-80
Rlayer-81
Slayer-82
Tlayer-83
Ulayer-84
Vlayer-85
W	variables
Xregularization_losses
Ytrainable_variables
Z	keras_api
[
signatures
 
 
 
h

\kernel
]bias
^	variables
_regularization_losses
`trainable_variables
a	keras_api
 
 
h

bkernel
cbias
d	variables
eregularization_losses
ftrainable_variables
g	keras_api
h

hkernel
ibias
j	variables
kregularization_losses
ltrainable_variables
m	keras_api
h

nkernel
obias
p	variables
qregularization_losses
rtrainable_variables
s	keras_api

t	keras_api

u	keras_api

v	keras_api

w	keras_api

x	keras_api

y	keras_api
h

zkernel
{bias
|	variables
}regularization_losses
~trainable_variables
	keras_api
 

А	keras_api

Б	keras_api

В	keras_api

Г	keras_api
n
Дkernel
	Еbias
Ж	variables
Зregularization_losses
Иtrainable_variables
Й	keras_api

К	keras_api

Л	keras_api

М	keras_api

Н	keras_api

О	keras_api

П	keras_api

Р	keras_api

С	keras_api

Т	keras_api

У	keras_api
n
Фkernel
	Хbias
Ц	variables
Чregularization_losses
Шtrainable_variables
Щ	keras_api
n
Ъkernel
	Ыbias
Ь	variables
Эregularization_losses
Юtrainable_variables
Я	keras_api

а	keras_api

б	keras_api
n
вkernel
	гbias
д	variables
еregularization_losses
жtrainable_variables
з	keras_api
n
иkernel
	йbias
к	variables
лregularization_losses
мtrainable_variables
н	keras_api

о	keras_api

п	keras_api

░	keras_api

▒	keras_api

▓	keras_api

│	keras_api

┤	keras_api

╡	keras_api

╢	keras_api

╖	keras_api

╕	keras_api

╣	keras_api

║	keras_api

╗	keras_api

╝	keras_api

╜	keras_api
n
╛kernel
	┐bias
└	variables
┴regularization_losses
┬trainable_variables
├	keras_api
n
─kernel
	┼bias
╞	variables
╟regularization_losses
╚trainable_variables
╔	keras_api

╩	keras_api

╦	keras_api

╠	keras_api

═	keras_api

╬	keras_api

╧	keras_api

╨	keras_api

╤	keras_api

╥	keras_api

╙	keras_api

╘	keras_api

╒	keras_api

╓	keras_api

╫	keras_api

╪	keras_api

┘	keras_api

┌	keras_api

█	keras_api

▄	keras_api

▌	keras_api

▐	keras_api

▀	keras_api

р	keras_api

с	keras_api

т	keras_api

у	keras_api

ф	keras_api

х	keras_api

ц	keras_api

ч	keras_api
─
\0
]1
b2
c3
h4
i5
n6
o7
z8
{9
Д10
Е11
Ф12
Х13
Ъ14
Ы15
в16
г17
и18
й19
╛20
┐21
─22
┼23
 
─
\0
]1
b2
c3
h4
i5
n6
o7
z8
{9
Д10
Е11
Ф12
Х13
Ъ14
Ы15
в16
г17
и18
й19
╛20
┐21
─22
┼23
▓
шmetrics
W	variables
Xregularization_losses
щlayers
ъlayer_metrics
Ytrainable_variables
ыnon_trainable_variables
 ьlayer_regularization_losses
 
`^
VARIABLE_VALUEencoder2/fcn1/kernel6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUE
\Z
VARIABLE_VALUEencoder2/fcn1/bias4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUE

\0
]1
 

\0
]1
▓
эlayer_metrics
юmetrics
^	variables
яlayers
_regularization_losses
 Ёlayer_regularization_losses
`trainable_variables
ёnon_trainable_variables
`^
VARIABLE_VALUEencoder1/fcn1/kernel6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUE
\Z
VARIABLE_VALUEencoder1/fcn1/bias4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUE

b0
c1
 

b0
c1
▓
Єlayer_metrics
єmetrics
d	variables
Їlayers
eregularization_losses
 їlayer_regularization_losses
ftrainable_variables
Ўnon_trainable_variables
ec
VARIABLE_VALUEencoder1/fcn3_gene/kernel6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUE
a_
VARIABLE_VALUEencoder1/fcn3_gene/bias4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUE

h0
i1
 

h0
i1
▓
ўlayer_metrics
°metrics
j	variables
∙layers
kregularization_losses
 ·layer_regularization_losses
ltrainable_variables
√non_trainable_variables
ec
VARIABLE_VALUEencoder2/fcn2/mean/kernel6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUE
a_
VARIABLE_VALUEencoder2/fcn2/mean/bias4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUE

n0
o1
 

n0
o1
▓
№layer_metrics
¤metrics
p	variables
■layers
qregularization_losses
  layer_regularization_losses
rtrainable_variables
Аnon_trainable_variables
 
 
 
 
 
 
ec
VARIABLE_VALUEencoder1/fcn2/mean/kernel6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUE
a_
VARIABLE_VALUEencoder1/fcn2/mean/bias4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUE

z0
{1
 

z0
{1
▓
Бlayer_metrics
Вmetrics
|	variables
Гlayers
}regularization_losses
 Дlayer_regularization_losses
~trainable_variables
Еnon_trainable_variables
 
 
 
 
fd
VARIABLE_VALUEdecoder1/layer_gene/kernel6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUE
b`
VARIABLE_VALUEdecoder1/layer_gene/bias4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUE

Д0
Е1
 

Д0
Е1
╡
Жlayer_metrics
Зmetrics
Ж	variables
Иlayers
Зregularization_losses
 Йlayer_regularization_losses
Иtrainable_variables
Кnon_trainable_variables
 
 
 
 
 
 
 
 
 
 
`^
VARIABLE_VALUEdecoder1/fcn1/kernel6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUE
\Z
VARIABLE_VALUEdecoder1/fcn1/bias4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUE

Ф0
Х1
 

Ф0
Х1
╡
Лlayer_metrics
Мmetrics
Ц	variables
Нlayers
Чregularization_losses
 Оlayer_regularization_losses
Шtrainable_variables
Пnon_trainable_variables
`^
VARIABLE_VALUEdecoder2/fcn1/kernel6layer_with_weights-7/kernel/.ATTRIBUTES/VARIABLE_VALUE
\Z
VARIABLE_VALUEdecoder2/fcn1/bias4layer_with_weights-7/bias/.ATTRIBUTES/VARIABLE_VALUE

Ъ0
Ы1
 

Ъ0
Ы1
╡
Рlayer_metrics
Сmetrics
Ь	variables
Тlayers
Эregularization_losses
 Уlayer_regularization_losses
Юtrainable_variables
Фnon_trainable_variables
 
 
`^
VARIABLE_VALUEdecoder1/fcn2/kernel6layer_with_weights-8/kernel/.ATTRIBUTES/VARIABLE_VALUE
\Z
VARIABLE_VALUEdecoder1/fcn2/bias4layer_with_weights-8/bias/.ATTRIBUTES/VARIABLE_VALUE

в0
г1
 

в0
г1
╡
Хlayer_metrics
Цmetrics
д	variables
Чlayers
еregularization_losses
 Шlayer_regularization_losses
жtrainable_variables
Щnon_trainable_variables
`^
VARIABLE_VALUEdecoder2/fcn2/kernel6layer_with_weights-9/kernel/.ATTRIBUTES/VARIABLE_VALUE
\Z
VARIABLE_VALUEdecoder2/fcn2/bias4layer_with_weights-9/bias/.ATTRIBUTES/VARIABLE_VALUE

и0
й1
 

и0
й1
╡
Ъlayer_metrics
Ыmetrics
к	variables
Ьlayers
лregularization_losses
 Эlayer_regularization_losses
мtrainable_variables
Юnon_trainable_variables
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
a_
VARIABLE_VALUEencoder1/fcn3/kernel7layer_with_weights-10/kernel/.ATTRIBUTES/VARIABLE_VALUE
][
VARIABLE_VALUEencoder1/fcn3/bias5layer_with_weights-10/bias/.ATTRIBUTES/VARIABLE_VALUE

╛0
┐1
 

╛0
┐1
╡
Яlayer_metrics
аmetrics
└	variables
бlayers
┴regularization_losses
 вlayer_regularization_losses
┬trainable_variables
гnon_trainable_variables
a_
VARIABLE_VALUEencoder2/fcn3/kernel7layer_with_weights-11/kernel/.ATTRIBUTES/VARIABLE_VALUE
][
VARIABLE_VALUEencoder2/fcn3/bias5layer_with_weights-11/bias/.ATTRIBUTES/VARIABLE_VALUE

─0
┼1
 

─0
┼1
╡
дlayer_metrics
еmetrics
╞	variables
жlayers
╟regularization_losses
 зlayer_regularization_losses
╚trainable_variables
иnon_trainable_variables
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
ж
0
1
2
3
4
5
6
7
	8

9
10
11
12
13
14
15
16
17
18
19
20
21
22
23
24
25
26
27
28
29
30
 31
!32
"33
#34
$35
%36
&37
'38
(39
)40
*41
+42
,43
-44
.45
/46
047
148
249
350
451
552
653
754
855
956
:57
;58
<59
=60
>61
?62
@63
A64
B65
C66
D67
E68
F69
G70
H71
I72
J73
K74
L75
M76
N77
O78
P79
Q80
R81
S82
T83
U84
V85
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
|
serving_default_input_1Placeholder*(
_output_shapes
:         А*
dtype0*
shape:         А
|
serving_default_input_2Placeholder*(
_output_shapes
:         А*
dtype0*
shape:         А
|
serving_default_input_3Placeholder*(
_output_shapes
:         А*
dtype0*
shape:         А
|
serving_default_input_4Placeholder*(
_output_shapes
:         А*
dtype0*
shape:         А
|
serving_default_input_5Placeholder*(
_output_shapes
:         А*
dtype0*
shape:         А
|
serving_default_input_6Placeholder*(
_output_shapes
:         А*
dtype0*
shape:         А
║
StatefulPartitionedCallStatefulPartitionedCallserving_default_input_1serving_default_input_2serving_default_input_3serving_default_input_4serving_default_input_5serving_default_input_6encoder2/fcn1/kernelencoder2/fcn1/biasencoder2/fcn2/mean/kernelencoder2/fcn2/mean/biasencoder1/fcn3_gene/kernelencoder1/fcn3_gene/biasencoder1/fcn1/kernelencoder1/fcn1/biasencoder1/fcn2/mean/kernelencoder1/fcn2/mean/biasdecoder1/layer_gene/kerneldecoder1/layer_gene/biasdecoder2/fcn1/kerneldecoder2/fcn1/biasdecoder1/fcn1/kerneldecoder1/fcn1/biasdecoder2/fcn2/kerneldecoder2/fcn2/biasdecoder1/fcn2/kerneldecoder1/fcn2/biasencoder2/fcn3/kernelencoder2/fcn3/biasencoder1/fcn3/kernelencoder1/fcn3/biasConst**
Tin#
!2*
Tout
2*
_collective_manager_ids
 *K
_output_shapes9
7:         P:         P: :         *:
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8В */
f*R(
&__inference_signature_wrapper_40050805
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
╝

StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename(encoder2/fcn1/kernel/Read/ReadVariableOp&encoder2/fcn1/bias/Read/ReadVariableOp(encoder1/fcn1/kernel/Read/ReadVariableOp&encoder1/fcn1/bias/Read/ReadVariableOp-encoder1/fcn3_gene/kernel/Read/ReadVariableOp+encoder1/fcn3_gene/bias/Read/ReadVariableOp-encoder2/fcn2/mean/kernel/Read/ReadVariableOp+encoder2/fcn2/mean/bias/Read/ReadVariableOp-encoder1/fcn2/mean/kernel/Read/ReadVariableOp+encoder1/fcn2/mean/bias/Read/ReadVariableOp.decoder1/layer_gene/kernel/Read/ReadVariableOp,decoder1/layer_gene/bias/Read/ReadVariableOp(decoder1/fcn1/kernel/Read/ReadVariableOp&decoder1/fcn1/bias/Read/ReadVariableOp(decoder2/fcn1/kernel/Read/ReadVariableOp&decoder2/fcn1/bias/Read/ReadVariableOp(decoder1/fcn2/kernel/Read/ReadVariableOp&decoder1/fcn2/bias/Read/ReadVariableOp(decoder2/fcn2/kernel/Read/ReadVariableOp&decoder2/fcn2/bias/Read/ReadVariableOp(encoder1/fcn3/kernel/Read/ReadVariableOp&encoder1/fcn3/bias/Read/ReadVariableOp(encoder2/fcn3/kernel/Read/ReadVariableOp&encoder2/fcn3/bias/Read/ReadVariableOpConst_1*%
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В **
f%R#
!__inference__traced_save_40051734
╒
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenameencoder2/fcn1/kernelencoder2/fcn1/biasencoder1/fcn1/kernelencoder1/fcn1/biasencoder1/fcn3_gene/kernelencoder1/fcn3_gene/biasencoder2/fcn2/mean/kernelencoder2/fcn2/mean/biasencoder1/fcn2/mean/kernelencoder1/fcn2/mean/biasdecoder1/layer_gene/kerneldecoder1/layer_gene/biasdecoder1/fcn1/kerneldecoder1/fcn1/biasdecoder2/fcn1/kerneldecoder2/fcn1/biasdecoder1/fcn2/kerneldecoder1/fcn2/biasdecoder2/fcn2/kerneldecoder2/fcn2/biasencoder1/fcn3/kernelencoder1/fcn3/biasencoder2/fcn3/kernelencoder2/fcn3/bias*$
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *-
f(R&
$__inference__traced_restore_40051816■╫
─

В
Q__inference_decoder1/layer_gene_layer_call_and_return_conditional_losses_40049582

inputs0
matmul_readvariableop_resource:dd-
biasadd_readvariableop_resource:d
identityИвBiasAdd/ReadVariableOpвMatMul/ReadVariableOpН
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:dd*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         d2
MatMulМ
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:d*
dtype02
BiasAdd/ReadVariableOpБ
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         d2	
BiasAdd^
	LeakyRelu	LeakyReluBiasAdd:output:0*'
_output_shapes
:         d2
	LeakyReluЬ
IdentityIdentityLeakyRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:         d2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:         d: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:         d
 
_user_specified_nameinputs
├

Б
P__inference_encoder2/fcn2/mean_layer_call_and_return_conditional_losses_40051465

inputs0
matmul_readvariableop_resource:x`-
biasadd_readvariableop_resource:`
identityИвBiasAdd/ReadVariableOpвMatMul/ReadVariableOpН
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:x`*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         `2
MatMulМ
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:`*
dtype02
BiasAdd/ReadVariableOpБ
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         `2	
BiasAdd^
	LeakyRelu	LeakyReluBiasAdd:output:0*'
_output_shapes
:         `2
	LeakyReluЬ
IdentityIdentityLeakyRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:         `2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:         x: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:         x
 
_user_specified_nameinputs
т	
 
K__inference_decoder2/fcn2_layer_call_and_return_conditional_losses_40049656

inputs2
matmul_readvariableop_resource:
А	А.
biasadd_readvariableop_resource:	А
identityИвBiasAdd/ReadVariableOpвMatMul/ReadVariableOpП
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
А	А*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         А2
MatMulН
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:А*
dtype02
BiasAdd/ReadVariableOpВ
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         А2	
BiasAddЦ
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*(
_output_shapes
:         А2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:         А	: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:         А	
 
_user_specified_nameinputs
ъh
╥
$__inference__traced_restore_40051816
file_prefix8
%assignvariableop_encoder2_fcn1_kernel:	Аx3
%assignvariableop_1_encoder2_fcn1_bias:x:
'assignvariableop_2_encoder1_fcn1_kernel:	Аd3
%assignvariableop_3_encoder1_fcn1_bias:d?
,assignvariableop_4_encoder1_fcn3_gene_kernel:	Аd8
*assignvariableop_5_encoder1_fcn3_gene_bias:d>
,assignvariableop_6_encoder2_fcn2_mean_kernel:x`8
*assignvariableop_7_encoder2_fcn2_mean_bias:`>
,assignvariableop_8_encoder1_fcn2_mean_kernel:dd8
*assignvariableop_9_encoder1_fcn2_mean_bias:d@
.assignvariableop_10_decoder1_layer_gene_kernel:dd:
,assignvariableop_11_decoder1_layer_gene_bias:d;
(assignvariableop_12_decoder1_fcn1_kernel:	dА	5
&assignvariableop_13_decoder1_fcn1_bias:	А	;
(assignvariableop_14_decoder2_fcn1_kernel:	`А	5
&assignvariableop_15_decoder2_fcn1_bias:	А	<
(assignvariableop_16_decoder1_fcn2_kernel:
А	А5
&assignvariableop_17_decoder1_fcn2_bias:	А<
(assignvariableop_18_decoder2_fcn2_kernel:
А	А5
&assignvariableop_19_decoder2_fcn2_bias:	А:
(assignvariableop_20_encoder1_fcn3_kernel:dP4
&assignvariableop_21_encoder1_fcn3_bias:P:
(assignvariableop_22_encoder2_fcn3_kernel:`P4
&assignvariableop_23_encoder2_fcn3_bias:P
identity_25ИвAssignVariableOpвAssignVariableOp_1вAssignVariableOp_10вAssignVariableOp_11вAssignVariableOp_12вAssignVariableOp_13вAssignVariableOp_14вAssignVariableOp_15вAssignVariableOp_16вAssignVariableOp_17вAssignVariableOp_18вAssignVariableOp_19вAssignVariableOp_2вAssignVariableOp_20вAssignVariableOp_21вAssignVariableOp_22вAssignVariableOp_23вAssignVariableOp_3вAssignVariableOp_4вAssignVariableOp_5вAssignVariableOp_6вAssignVariableOp_7вAssignVariableOp_8вAssignVariableOp_9╙
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*▀

value╒
B╥
B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-7/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-7/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-8/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-8/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-9/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-9/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-10/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-10/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-11/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-11/bias/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
RestoreV2/tensor_names└
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*E
value<B:B B B B B B B B B B B B B B B B B B B B B B B B B 2
RestoreV2/shape_and_slicesи
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*x
_output_shapesf
d:::::::::::::::::::::::::*'
dtypes
22
	RestoreV2g
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:2

Identityд
AssignVariableOpAssignVariableOp%assignvariableop_encoder2_fcn1_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOpk

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:2

Identity_1к
AssignVariableOp_1AssignVariableOp%assignvariableop_1_encoder2_fcn1_biasIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_1k

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:2

Identity_2м
AssignVariableOp_2AssignVariableOp'assignvariableop_2_encoder1_fcn1_kernelIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_2k

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:2

Identity_3к
AssignVariableOp_3AssignVariableOp%assignvariableop_3_encoder1_fcn1_biasIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_3k

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:2

Identity_4▒
AssignVariableOp_4AssignVariableOp,assignvariableop_4_encoder1_fcn3_gene_kernelIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_4k

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:2

Identity_5п
AssignVariableOp_5AssignVariableOp*assignvariableop_5_encoder1_fcn3_gene_biasIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_5k

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:2

Identity_6▒
AssignVariableOp_6AssignVariableOp,assignvariableop_6_encoder2_fcn2_mean_kernelIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_6k

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:2

Identity_7п
AssignVariableOp_7AssignVariableOp*assignvariableop_7_encoder2_fcn2_mean_biasIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_7k

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:2

Identity_8▒
AssignVariableOp_8AssignVariableOp,assignvariableop_8_encoder1_fcn2_mean_kernelIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_8k

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:2

Identity_9п
AssignVariableOp_9AssignVariableOp*assignvariableop_9_encoder1_fcn2_mean_biasIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_9n
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:2
Identity_10╢
AssignVariableOp_10AssignVariableOp.assignvariableop_10_decoder1_layer_gene_kernelIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_10n
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:2
Identity_11┤
AssignVariableOp_11AssignVariableOp,assignvariableop_11_decoder1_layer_gene_biasIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_11n
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:2
Identity_12░
AssignVariableOp_12AssignVariableOp(assignvariableop_12_decoder1_fcn1_kernelIdentity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_12n
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:2
Identity_13о
AssignVariableOp_13AssignVariableOp&assignvariableop_13_decoder1_fcn1_biasIdentity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_13n
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:2
Identity_14░
AssignVariableOp_14AssignVariableOp(assignvariableop_14_decoder2_fcn1_kernelIdentity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_14n
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:2
Identity_15о
AssignVariableOp_15AssignVariableOp&assignvariableop_15_decoder2_fcn1_biasIdentity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_15n
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:2
Identity_16░
AssignVariableOp_16AssignVariableOp(assignvariableop_16_decoder1_fcn2_kernelIdentity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_16n
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:2
Identity_17о
AssignVariableOp_17AssignVariableOp&assignvariableop_17_decoder1_fcn2_biasIdentity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_17n
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:2
Identity_18░
AssignVariableOp_18AssignVariableOp(assignvariableop_18_decoder2_fcn2_kernelIdentity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_18n
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:2
Identity_19о
AssignVariableOp_19AssignVariableOp&assignvariableop_19_decoder2_fcn2_biasIdentity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_19n
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:2
Identity_20░
AssignVariableOp_20AssignVariableOp(assignvariableop_20_encoder1_fcn3_kernelIdentity_20:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_20n
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:2
Identity_21о
AssignVariableOp_21AssignVariableOp&assignvariableop_21_encoder1_fcn3_biasIdentity_21:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_21n
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:2
Identity_22░
AssignVariableOp_22AssignVariableOp(assignvariableop_22_encoder2_fcn3_kernelIdentity_22:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_22n
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:2
Identity_23о
AssignVariableOp_23AssignVariableOp&assignvariableop_23_encoder2_fcn3_biasIdentity_23:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_239
NoOpNoOp"/device:CPU:0*
_output_shapes
 2
NoOpю
Identity_24Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: 2
Identity_24с
Identity_25IdentityIdentity_24:output:0^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9*
T0*
_output_shapes
: 2
Identity_25"#
identity_25Identity_25:output:0*E
_input_shapes4
2: : : : : : : : : : : : : : : : : : : : : : : : : 2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12*
AssignVariableOp_10AssignVariableOp_102*
AssignVariableOp_11AssignVariableOp_112*
AssignVariableOp_12AssignVariableOp_122*
AssignVariableOp_13AssignVariableOp_132*
AssignVariableOp_14AssignVariableOp_142*
AssignVariableOp_15AssignVariableOp_152*
AssignVariableOp_16AssignVariableOp_162*
AssignVariableOp_17AssignVariableOp_172*
AssignVariableOp_18AssignVariableOp_182*
AssignVariableOp_19AssignVariableOp_192(
AssignVariableOp_2AssignVariableOp_22*
AssignVariableOp_20AssignVariableOp_202*
AssignVariableOp_21AssignVariableOp_212*
AssignVariableOp_22AssignVariableOp_222*
AssignVariableOp_23AssignVariableOp_232(
AssignVariableOp_3AssignVariableOp_32(
AssignVariableOp_4AssignVariableOp_42(
AssignVariableOp_5AssignVariableOp_52(
AssignVariableOp_6AssignVariableOp_62(
AssignVariableOp_7AssignVariableOp_72(
AssignVariableOp_8AssignVariableOp_82(
AssignVariableOp_9AssignVariableOp_9:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
╞

■
K__inference_decoder2/fcn1_layer_call_and_return_conditional_losses_40049613

inputs1
matmul_readvariableop_resource:	`А	.
biasadd_readvariableop_resource:	А	
identityИвBiasAdd/ReadVariableOpвMatMul/ReadVariableOpО
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	`А	*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         А	2
MatMulН
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:А	*
dtype02
BiasAdd/ReadVariableOpВ
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         А	2	
BiasAdd_
	LeakyRelu	LeakyReluBiasAdd:output:0*(
_output_shapes
:         А	2
	LeakyReluЭ
IdentityIdentityLeakyRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*(
_output_shapes
:         А	2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:         `: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:         `
 
_user_specified_nameinputs
░
Я
0__inference_decoder1/fcn1_layer_call_fn_40051534

inputs
unknown:	dА	
	unknown_0:	А	
identityИвStatefulPartitionedCall 
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         А	*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *T
fORM
K__inference_decoder1/fcn1_layer_call_and_return_conditional_losses_400496302
StatefulPartitionedCallП
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:         А	2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:         d: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:         d
 
_user_specified_nameinputs
│
а
0__inference_decoder2/fcn2_layer_call_fn_40051592

inputs
unknown:
А	А
	unknown_0:	А
identityИвStatefulPartitionedCall 
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         А*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *T
fORM
K__inference_decoder2/fcn2_layer_call_and_return_conditional_losses_400496562
StatefulPartitionedCallП
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:         А2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:         А	: : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:         А	
 
_user_specified_nameinputs
┬п
Н
#__inference__wrapped_model_40049440
input_1
input_2
input_3
input_4
input_5
input_6E
2model_encoder2_fcn1_matmul_readvariableop_resource:	АxA
3model_encoder2_fcn1_biasadd_readvariableop_resource:xI
7model_encoder2_fcn2_mean_matmul_readvariableop_resource:x`F
8model_encoder2_fcn2_mean_biasadd_readvariableop_resource:`J
7model_encoder1_fcn3_gene_matmul_readvariableop_resource:	АdF
8model_encoder1_fcn3_gene_biasadd_readvariableop_resource:dE
2model_encoder1_fcn1_matmul_readvariableop_resource:	АdA
3model_encoder1_fcn1_biasadd_readvariableop_resource:dI
7model_encoder1_fcn2_mean_matmul_readvariableop_resource:ddF
8model_encoder1_fcn2_mean_biasadd_readvariableop_resource:dJ
8model_decoder1_layer_gene_matmul_readvariableop_resource:ddG
9model_decoder1_layer_gene_biasadd_readvariableop_resource:dE
2model_decoder2_fcn1_matmul_readvariableop_resource:	`А	B
3model_decoder2_fcn1_biasadd_readvariableop_resource:	А	E
2model_decoder1_fcn1_matmul_readvariableop_resource:	dА	B
3model_decoder1_fcn1_biasadd_readvariableop_resource:	А	F
2model_decoder2_fcn2_matmul_readvariableop_resource:
А	АB
3model_decoder2_fcn2_biasadd_readvariableop_resource:	АF
2model_decoder1_fcn2_matmul_readvariableop_resource:
А	АB
3model_decoder1_fcn2_biasadd_readvariableop_resource:	АD
2model_encoder2_fcn3_matmul_readvariableop_resource:`PA
3model_encoder2_fcn3_biasadd_readvariableop_resource:PD
2model_encoder1_fcn3_matmul_readvariableop_resource:dPA
3model_encoder1_fcn3_biasadd_readvariableop_resource:P
model_40049411
identity

identity_1

identity_2

identity_3Ив*model/decoder1/fcn1/BiasAdd/ReadVariableOpв,model/decoder1/fcn1/BiasAdd_1/ReadVariableOpв)model/decoder1/fcn1/MatMul/ReadVariableOpв+model/decoder1/fcn1/MatMul_1/ReadVariableOpв*model/decoder1/fcn2/BiasAdd/ReadVariableOpв,model/decoder1/fcn2/BiasAdd_1/ReadVariableOpв)model/decoder1/fcn2/MatMul/ReadVariableOpв+model/decoder1/fcn2/MatMul_1/ReadVariableOpв0model/decoder1/layer_gene/BiasAdd/ReadVariableOpв2model/decoder1/layer_gene/BiasAdd_1/ReadVariableOpв/model/decoder1/layer_gene/MatMul/ReadVariableOpв1model/decoder1/layer_gene/MatMul_1/ReadVariableOpв*model/decoder2/fcn1/BiasAdd/ReadVariableOpв,model/decoder2/fcn1/BiasAdd_1/ReadVariableOpв)model/decoder2/fcn1/MatMul/ReadVariableOpв+model/decoder2/fcn1/MatMul_1/ReadVariableOpв*model/decoder2/fcn2/BiasAdd/ReadVariableOpв,model/decoder2/fcn2/BiasAdd_1/ReadVariableOpв)model/decoder2/fcn2/MatMul/ReadVariableOpв+model/decoder2/fcn2/MatMul_1/ReadVariableOpв*model/encoder1/fcn1/BiasAdd/ReadVariableOpв,model/encoder1/fcn1/BiasAdd_1/ReadVariableOpв)model/encoder1/fcn1/MatMul/ReadVariableOpв+model/encoder1/fcn1/MatMul_1/ReadVariableOpв/model/encoder1/fcn2/mean/BiasAdd/ReadVariableOpв1model/encoder1/fcn2/mean/BiasAdd_1/ReadVariableOpв.model/encoder1/fcn2/mean/MatMul/ReadVariableOpв0model/encoder1/fcn2/mean/MatMul_1/ReadVariableOpв*model/encoder1/fcn3/BiasAdd/ReadVariableOpв)model/encoder1/fcn3/MatMul/ReadVariableOpв/model/encoder1/fcn3_gene/BiasAdd/ReadVariableOpв1model/encoder1/fcn3_gene/BiasAdd_1/ReadVariableOpв.model/encoder1/fcn3_gene/MatMul/ReadVariableOpв0model/encoder1/fcn3_gene/MatMul_1/ReadVariableOpв*model/encoder2/fcn1/BiasAdd/ReadVariableOpв,model/encoder2/fcn1/BiasAdd_1/ReadVariableOpв)model/encoder2/fcn1/MatMul/ReadVariableOpв+model/encoder2/fcn1/MatMul_1/ReadVariableOpв/model/encoder2/fcn2/mean/BiasAdd/ReadVariableOpв1model/encoder2/fcn2/mean/BiasAdd_1/ReadVariableOpв.model/encoder2/fcn2/mean/MatMul/ReadVariableOpв0model/encoder2/fcn2/mean/MatMul_1/ReadVariableOpв*model/encoder2/fcn3/BiasAdd/ReadVariableOpв)model/encoder2/fcn3/MatMul/ReadVariableOp╩
)model/encoder2/fcn1/MatMul/ReadVariableOpReadVariableOp2model_encoder2_fcn1_matmul_readvariableop_resource*
_output_shapes
:	Аx*
dtype02+
)model/encoder2/fcn1/MatMul/ReadVariableOp░
model/encoder2/fcn1/MatMulMatMulinput_21model/encoder2/fcn1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         x2
model/encoder2/fcn1/MatMul╚
*model/encoder2/fcn1/BiasAdd/ReadVariableOpReadVariableOp3model_encoder2_fcn1_biasadd_readvariableop_resource*
_output_shapes
:x*
dtype02,
*model/encoder2/fcn1/BiasAdd/ReadVariableOp╤
model/encoder2/fcn1/BiasAddBiasAdd$model/encoder2/fcn1/MatMul:product:02model/encoder2/fcn1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         x2
model/encoder2/fcn1/BiasAddФ
model/encoder2/fcn1/ReluRelu$model/encoder2/fcn1/BiasAdd:output:0*
T0*'
_output_shapes
:         x2
model/encoder2/fcn1/Relu╪
.model/encoder2/fcn2/mean/MatMul/ReadVariableOpReadVariableOp7model_encoder2_fcn2_mean_matmul_readvariableop_resource*
_output_shapes

:x`*
dtype020
.model/encoder2/fcn2/mean/MatMul/ReadVariableOp▐
model/encoder2/fcn2/mean/MatMulMatMul&model/encoder2/fcn1/Relu:activations:06model/encoder2/fcn2/mean/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         `2!
model/encoder2/fcn2/mean/MatMul╫
/model/encoder2/fcn2/mean/BiasAdd/ReadVariableOpReadVariableOp8model_encoder2_fcn2_mean_biasadd_readvariableop_resource*
_output_shapes
:`*
dtype021
/model/encoder2/fcn2/mean/BiasAdd/ReadVariableOpх
 model/encoder2/fcn2/mean/BiasAddBiasAdd)model/encoder2/fcn2/mean/MatMul:product:07model/encoder2/fcn2/mean/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         `2"
 model/encoder2/fcn2/mean/BiasAddй
"model/encoder2/fcn2/mean/LeakyRelu	LeakyRelu)model/encoder2/fcn2/mean/BiasAdd:output:0*'
_output_shapes
:         `2$
"model/encoder2/fcn2/mean/LeakyRelu┘
.model/encoder1/fcn3_gene/MatMul/ReadVariableOpReadVariableOp7model_encoder1_fcn3_gene_matmul_readvariableop_resource*
_output_shapes
:	Аd*
dtype020
.model/encoder1/fcn3_gene/MatMul/ReadVariableOp┐
model/encoder1/fcn3_gene/MatMulMatMulinput_46model/encoder1/fcn3_gene/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         d2!
model/encoder1/fcn3_gene/MatMul╫
/model/encoder1/fcn3_gene/BiasAdd/ReadVariableOpReadVariableOp8model_encoder1_fcn3_gene_biasadd_readvariableop_resource*
_output_shapes
:d*
dtype021
/model/encoder1/fcn3_gene/BiasAdd/ReadVariableOpх
 model/encoder1/fcn3_gene/BiasAddBiasAdd)model/encoder1/fcn3_gene/MatMul:product:07model/encoder1/fcn3_gene/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         d2"
 model/encoder1/fcn3_gene/BiasAdd╩
)model/encoder1/fcn1/MatMul/ReadVariableOpReadVariableOp2model_encoder1_fcn1_matmul_readvariableop_resource*
_output_shapes
:	Аd*
dtype02+
)model/encoder1/fcn1/MatMul/ReadVariableOp░
model/encoder1/fcn1/MatMulMatMulinput_31model/encoder1/fcn1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         d2
model/encoder1/fcn1/MatMul╚
*model/encoder1/fcn1/BiasAdd/ReadVariableOpReadVariableOp3model_encoder1_fcn1_biasadd_readvariableop_resource*
_output_shapes
:d*
dtype02,
*model/encoder1/fcn1/BiasAdd/ReadVariableOp╤
model/encoder1/fcn1/BiasAddBiasAdd$model/encoder1/fcn1/MatMul:product:02model/encoder1/fcn1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         d2
model/encoder1/fcn1/BiasAddФ
model/encoder1/fcn1/ReluRelu$model/encoder1/fcn1/BiasAdd:output:0*
T0*'
_output_shapes
:         d2
model/encoder1/fcn1/Relu▌
0model/encoder1/fcn3_gene/MatMul_1/ReadVariableOpReadVariableOp7model_encoder1_fcn3_gene_matmul_readvariableop_resource*
_output_shapes
:	Аd*
dtype022
0model/encoder1/fcn3_gene/MatMul_1/ReadVariableOp┼
!model/encoder1/fcn3_gene/MatMul_1MatMulinput_68model/encoder1/fcn3_gene/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:         d2#
!model/encoder1/fcn3_gene/MatMul_1█
1model/encoder1/fcn3_gene/BiasAdd_1/ReadVariableOpReadVariableOp8model_encoder1_fcn3_gene_biasadd_readvariableop_resource*
_output_shapes
:d*
dtype023
1model/encoder1/fcn3_gene/BiasAdd_1/ReadVariableOpэ
"model/encoder1/fcn3_gene/BiasAdd_1BiasAdd+model/encoder1/fcn3_gene/MatMul_1:product:09model/encoder1/fcn3_gene/BiasAdd_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:         d2$
"model/encoder1/fcn3_gene/BiasAdd_1╬
+model/encoder1/fcn1/MatMul_1/ReadVariableOpReadVariableOp2model_encoder1_fcn1_matmul_readvariableop_resource*
_output_shapes
:	Аd*
dtype02-
+model/encoder1/fcn1/MatMul_1/ReadVariableOp╢
model/encoder1/fcn1/MatMul_1MatMulinput_13model/encoder1/fcn1/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:         d2
model/encoder1/fcn1/MatMul_1╠
,model/encoder1/fcn1/BiasAdd_1/ReadVariableOpReadVariableOp3model_encoder1_fcn1_biasadd_readvariableop_resource*
_output_shapes
:d*
dtype02.
,model/encoder1/fcn1/BiasAdd_1/ReadVariableOp┘
model/encoder1/fcn1/BiasAdd_1BiasAdd&model/encoder1/fcn1/MatMul_1:product:04model/encoder1/fcn1/BiasAdd_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:         d2
model/encoder1/fcn1/BiasAdd_1Ъ
model/encoder1/fcn1/Relu_1Relu&model/encoder1/fcn1/BiasAdd_1:output:0*
T0*'
_output_shapes
:         d2
model/encoder1/fcn1/Relu_1м
model/tf.math.square_5/SquareSquare0model/encoder2/fcn2/mean/LeakyRelu:activations:0*
T0*'
_output_shapes
:         `2
model/tf.math.square_5/Squareм
model/tf.math.square_4/SquareSquare0model/encoder2/fcn2/mean/LeakyRelu:activations:0*
T0*'
_output_shapes
:         `2
model/tf.math.square_4/SquareД
model/tf.math.square_3/SquareSquareinput_2*
T0*(
_output_shapes
:         А2
model/tf.math.square_3/SquareД
model/tf.math.square_2/SquareSquareinput_2*
T0*(
_output_shapes
:         А2
model/tf.math.square_2/Square╚
model/tf.math.subtract_3/SubSub&model/encoder1/fcn1/Relu:activations:0)model/encoder1/fcn3_gene/BiasAdd:output:0*
T0*'
_output_shapes
:         d2
model/tf.math.subtract_3/Sub╚
model/tf.math.subtract/SubSub(model/encoder1/fcn1/Relu_1:activations:0+model/encoder1/fcn3_gene/BiasAdd_1:output:0*
T0*'
_output_shapes
:         d2
model/tf.math.subtract/Subп
0model/tf.math.reduce_sum_3/Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
valueB :
         22
0model/tf.math.reduce_sum_3/Sum/reduction_indicesш
model/tf.math.reduce_sum_3/SumSum!model/tf.math.square_5/Square:y:09model/tf.math.reduce_sum_3/Sum/reduction_indices:output:0*
T0*'
_output_shapes
:         *
	keep_dims(2 
model/tf.math.reduce_sum_3/Sumп
0model/tf.math.reduce_sum_2/Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
valueB :
         22
0model/tf.math.reduce_sum_2/Sum/reduction_indicesш
model/tf.math.reduce_sum_2/SumSum!model/tf.math.square_4/Square:y:09model/tf.math.reduce_sum_2/Sum/reduction_indices:output:0*
T0*'
_output_shapes
:         *
	keep_dims(2 
model/tf.math.reduce_sum_2/Sumп
0model/tf.math.reduce_sum_1/Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
valueB :
         22
0model/tf.math.reduce_sum_1/Sum/reduction_indicesш
model/tf.math.reduce_sum_1/SumSum!model/tf.math.square_3/Square:y:09model/tf.math.reduce_sum_1/Sum/reduction_indices:output:0*
T0*'
_output_shapes
:         *
	keep_dims(2 
model/tf.math.reduce_sum_1/Sumл
.model/tf.math.reduce_sum/Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
valueB :
         20
.model/tf.math.reduce_sum/Sum/reduction_indicesт
model/tf.math.reduce_sum/SumSum!model/tf.math.square_2/Square:y:07model/tf.math.reduce_sum/Sum/reduction_indices:output:0*
T0*'
_output_shapes
:         *
	keep_dims(2
model/tf.math.reduce_sum/Sum╪
.model/encoder1/fcn2/mean/MatMul/ReadVariableOpReadVariableOp7model_encoder1_fcn2_mean_matmul_readvariableop_resource*
_output_shapes

:dd*
dtype020
.model/encoder1/fcn2/mean/MatMul/ReadVariableOp╪
model/encoder1/fcn2/mean/MatMulMatMul model/tf.math.subtract_3/Sub:z:06model/encoder1/fcn2/mean/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         d2!
model/encoder1/fcn2/mean/MatMul╫
/model/encoder1/fcn2/mean/BiasAdd/ReadVariableOpReadVariableOp8model_encoder1_fcn2_mean_biasadd_readvariableop_resource*
_output_shapes
:d*
dtype021
/model/encoder1/fcn2/mean/BiasAdd/ReadVariableOpх
 model/encoder1/fcn2/mean/BiasAddBiasAdd)model/encoder1/fcn2/mean/MatMul:product:07model/encoder1/fcn2/mean/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         d2"
 model/encoder1/fcn2/mean/BiasAddй
"model/encoder1/fcn2/mean/LeakyRelu	LeakyRelu)model/encoder1/fcn2/mean/BiasAdd:output:0*'
_output_shapes
:         d2$
"model/encoder1/fcn2/mean/LeakyRelu▄
0model/encoder1/fcn2/mean/MatMul_1/ReadVariableOpReadVariableOp7model_encoder1_fcn2_mean_matmul_readvariableop_resource*
_output_shapes

:dd*
dtype022
0model/encoder1/fcn2/mean/MatMul_1/ReadVariableOp▄
!model/encoder1/fcn2/mean/MatMul_1MatMulmodel/tf.math.subtract/Sub:z:08model/encoder1/fcn2/mean/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:         d2#
!model/encoder1/fcn2/mean/MatMul_1█
1model/encoder1/fcn2/mean/BiasAdd_1/ReadVariableOpReadVariableOp8model_encoder1_fcn2_mean_biasadd_readvariableop_resource*
_output_shapes
:d*
dtype023
1model/encoder1/fcn2/mean/BiasAdd_1/ReadVariableOpэ
"model/encoder1/fcn2/mean/BiasAdd_1BiasAdd+model/encoder1/fcn2/mean/MatMul_1:product:09model/encoder1/fcn2/mean/BiasAdd_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:         d2$
"model/encoder1/fcn2/mean/BiasAdd_1п
$model/encoder1/fcn2/mean/LeakyRelu_1	LeakyRelu+model/encoder1/fcn2/mean/BiasAdd_1:output:0*'
_output_shapes
:         d2&
$model/encoder1/fcn2/mean/LeakyRelu_1Щ
model/tf.math.sqrt_2/SqrtSqrt'model/tf.math.reduce_sum_2/Sum:output:0*
T0*'
_output_shapes
:         2
model/tf.math.sqrt_2/SqrtЩ
model/tf.math.sqrt_3/SqrtSqrt'model/tf.math.reduce_sum_3/Sum:output:0*
T0*'
_output_shapes
:         2
model/tf.math.sqrt_3/SqrtУ
model/tf.math.sqrt/SqrtSqrt%model/tf.math.reduce_sum/Sum:output:0*
T0*'
_output_shapes
:         2
model/tf.math.sqrt/SqrtЩ
model/tf.math.sqrt_1/SqrtSqrt'model/tf.math.reduce_sum_1/Sum:output:0*
T0*'
_output_shapes
:         2
model/tf.math.sqrt_1/Sqrt╬
+model/encoder2/fcn1/MatMul_1/ReadVariableOpReadVariableOp2model_encoder2_fcn1_matmul_readvariableop_resource*
_output_shapes
:	Аx*
dtype02-
+model/encoder2/fcn1/MatMul_1/ReadVariableOp╢
model/encoder2/fcn1/MatMul_1MatMulinput_53model/encoder2/fcn1/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:         x2
model/encoder2/fcn1/MatMul_1╠
,model/encoder2/fcn1/BiasAdd_1/ReadVariableOpReadVariableOp3model_encoder2_fcn1_biasadd_readvariableop_resource*
_output_shapes
:x*
dtype02.
,model/encoder2/fcn1/BiasAdd_1/ReadVariableOp┘
model/encoder2/fcn1/BiasAdd_1BiasAdd&model/encoder2/fcn1/MatMul_1:product:04model/encoder2/fcn1/BiasAdd_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:         x2
model/encoder2/fcn1/BiasAdd_1Ъ
model/encoder2/fcn1/Relu_1Relu&model/encoder2/fcn1/BiasAdd_1:output:0*
T0*'
_output_shapes
:         x2
model/encoder2/fcn1/Relu_1█
/model/decoder1/layer_gene/MatMul/ReadVariableOpReadVariableOp8model_decoder1_layer_gene_matmul_readvariableop_resource*
_output_shapes

:dd*
dtype021
/model/decoder1/layer_gene/MatMul/ReadVariableOpы
 model/decoder1/layer_gene/MatMulMatMul0model/encoder1/fcn2/mean/LeakyRelu:activations:07model/decoder1/layer_gene/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         d2"
 model/decoder1/layer_gene/MatMul┌
0model/decoder1/layer_gene/BiasAdd/ReadVariableOpReadVariableOp9model_decoder1_layer_gene_biasadd_readvariableop_resource*
_output_shapes
:d*
dtype022
0model/decoder1/layer_gene/BiasAdd/ReadVariableOpщ
!model/decoder1/layer_gene/BiasAddBiasAdd*model/decoder1/layer_gene/MatMul:product:08model/decoder1/layer_gene/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         d2#
!model/decoder1/layer_gene/BiasAddм
#model/decoder1/layer_gene/LeakyRelu	LeakyRelu*model/decoder1/layer_gene/BiasAdd:output:0*'
_output_shapes
:         d2%
#model/decoder1/layer_gene/LeakyRelu▀
1model/decoder1/layer_gene/MatMul_1/ReadVariableOpReadVariableOp8model_decoder1_layer_gene_matmul_readvariableop_resource*
_output_shapes

:dd*
dtype023
1model/decoder1/layer_gene/MatMul_1/ReadVariableOpє
"model/decoder1/layer_gene/MatMul_1MatMul2model/encoder1/fcn2/mean/LeakyRelu_1:activations:09model/decoder1/layer_gene/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:         d2$
"model/decoder1/layer_gene/MatMul_1▐
2model/decoder1/layer_gene/BiasAdd_1/ReadVariableOpReadVariableOp9model_decoder1_layer_gene_biasadd_readvariableop_resource*
_output_shapes
:d*
dtype024
2model/decoder1/layer_gene/BiasAdd_1/ReadVariableOpё
#model/decoder1/layer_gene/BiasAdd_1BiasAdd,model/decoder1/layer_gene/MatMul_1:product:0:model/decoder1/layer_gene/BiasAdd_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:         d2%
#model/decoder1/layer_gene/BiasAdd_1▓
%model/decoder1/layer_gene/LeakyRelu_1	LeakyRelu,model/decoder1/layer_gene/BiasAdd_1:output:0*'
_output_shapes
:         d2'
%model/decoder1/layer_gene/LeakyRelu_1■
model/tf.linalg.matmul_1/MatMulMatMul0model/encoder2/fcn2/mean/LeakyRelu:activations:00model/encoder2/fcn2/mean/LeakyRelu:activations:0*
T0*0
_output_shapes
:                  *
transpose_b(2!
model/tf.linalg.matmul_1/MatMul│
model/tf.math.multiply_3/MulMulmodel/tf.math.sqrt_2/Sqrt:y:0model/tf.math.sqrt_3/Sqrt:y:0*
T0*'
_output_shapes
:         2
model/tf.math.multiply_3/Mulи
model/tf.linalg.matmul/MatMulMatMulinput_2input_2*
T0*0
_output_shapes
:                  *
transpose_b(2
model/tf.linalg.matmul/MatMul▒
model/tf.math.multiply_2/MulMulmodel/tf.math.sqrt/Sqrt:y:0model/tf.math.sqrt_1/Sqrt:y:0*
T0*'
_output_shapes
:         2
model/tf.math.multiply_2/Mul▄
0model/encoder2/fcn2/mean/MatMul_1/ReadVariableOpReadVariableOp7model_encoder2_fcn2_mean_matmul_readvariableop_resource*
_output_shapes

:x`*
dtype022
0model/encoder2/fcn2/mean/MatMul_1/ReadVariableOpц
!model/encoder2/fcn2/mean/MatMul_1MatMul(model/encoder2/fcn1/Relu_1:activations:08model/encoder2/fcn2/mean/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:         `2#
!model/encoder2/fcn2/mean/MatMul_1█
1model/encoder2/fcn2/mean/BiasAdd_1/ReadVariableOpReadVariableOp8model_encoder2_fcn2_mean_biasadd_readvariableop_resource*
_output_shapes
:`*
dtype023
1model/encoder2/fcn2/mean/BiasAdd_1/ReadVariableOpэ
"model/encoder2/fcn2/mean/BiasAdd_1BiasAdd+model/encoder2/fcn2/mean/MatMul_1:product:09model/encoder2/fcn2/mean/BiasAdd_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:         `2$
"model/encoder2/fcn2/mean/BiasAdd_1п
$model/encoder2/fcn2/mean/LeakyRelu_1	LeakyRelu+model/encoder2/fcn2/mean/BiasAdd_1:output:0*'
_output_shapes
:         `2&
$model/encoder2/fcn2/mean/LeakyRelu_1с
"model/tf.__operators__.add_2/AddV2AddV21model/decoder1/layer_gene/LeakyRelu:activations:0)model/encoder1/fcn3_gene/BiasAdd:output:0*
T0*'
_output_shapes
:         d2$
"model/tf.__operators__.add_2/AddV2с
 model/tf.__operators__.add/AddV2AddV23model/decoder1/layer_gene/LeakyRelu_1:activations:0+model/encoder1/fcn3_gene/BiasAdd_1:output:0*
T0*'
_output_shapes
:         d2"
 model/tf.__operators__.add/AddV2╒
model/tf.math.truediv_1/truedivRealDiv)model/tf.linalg.matmul_1/MatMul:product:0 model/tf.math.multiply_3/Mul:z:0*
T0*0
_output_shapes
:                  2!
model/tf.math.truediv_1/truediv╧
model/tf.math.truediv/truedivRealDiv'model/tf.linalg.matmul/MatMul:product:0 model/tf.math.multiply_2/Mul:z:0*
T0*0
_output_shapes
:                  2
model/tf.math.truediv/truediv╩
)model/decoder2/fcn1/MatMul/ReadVariableOpReadVariableOp2model_decoder2_fcn1_matmul_readvariableop_resource*
_output_shapes
:	`А	*
dtype02+
)model/decoder2/fcn1/MatMul/ReadVariableOp▄
model/decoder2/fcn1/MatMulMatMul2model/encoder2/fcn2/mean/LeakyRelu_1:activations:01model/decoder2/fcn1/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         А	2
model/decoder2/fcn1/MatMul╔
*model/decoder2/fcn1/BiasAdd/ReadVariableOpReadVariableOp3model_decoder2_fcn1_biasadd_readvariableop_resource*
_output_shapes	
:А	*
dtype02,
*model/decoder2/fcn1/BiasAdd/ReadVariableOp╥
model/decoder2/fcn1/BiasAddBiasAdd$model/decoder2/fcn1/MatMul:product:02model/decoder2/fcn1/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         А	2
model/decoder2/fcn1/BiasAddЫ
model/decoder2/fcn1/LeakyRelu	LeakyRelu$model/decoder2/fcn1/BiasAdd:output:0*(
_output_shapes
:         А	2
model/decoder2/fcn1/LeakyRelu╩
)model/decoder1/fcn1/MatMul/ReadVariableOpReadVariableOp2model_decoder1_fcn1_matmul_readvariableop_resource*
_output_shapes
:	dА	*
dtype02+
)model/decoder1/fcn1/MatMul/ReadVariableOp╨
model/decoder1/fcn1/MatMulMatMul&model/tf.__operators__.add_2/AddV2:z:01model/decoder1/fcn1/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         А	2
model/decoder1/fcn1/MatMul╔
*model/decoder1/fcn1/BiasAdd/ReadVariableOpReadVariableOp3model_decoder1_fcn1_biasadd_readvariableop_resource*
_output_shapes	
:А	*
dtype02,
*model/decoder1/fcn1/BiasAdd/ReadVariableOp╥
model/decoder1/fcn1/BiasAddBiasAdd$model/decoder1/fcn1/MatMul:product:02model/decoder1/fcn1/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         А	2
model/decoder1/fcn1/BiasAddЫ
model/decoder1/fcn1/LeakyRelu	LeakyRelu$model/decoder1/fcn1/BiasAdd:output:0*(
_output_shapes
:         А	2
model/decoder1/fcn1/LeakyRelu╬
+model/decoder2/fcn1/MatMul_1/ReadVariableOpReadVariableOp2model_decoder2_fcn1_matmul_readvariableop_resource*
_output_shapes
:	`А	*
dtype02-
+model/decoder2/fcn1/MatMul_1/ReadVariableOpр
model/decoder2/fcn1/MatMul_1MatMul0model/encoder2/fcn2/mean/LeakyRelu:activations:03model/decoder2/fcn1/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:         А	2
model/decoder2/fcn1/MatMul_1═
,model/decoder2/fcn1/BiasAdd_1/ReadVariableOpReadVariableOp3model_decoder2_fcn1_biasadd_readvariableop_resource*
_output_shapes	
:А	*
dtype02.
,model/decoder2/fcn1/BiasAdd_1/ReadVariableOp┌
model/decoder2/fcn1/BiasAdd_1BiasAdd&model/decoder2/fcn1/MatMul_1:product:04model/decoder2/fcn1/BiasAdd_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:         А	2
model/decoder2/fcn1/BiasAdd_1б
model/decoder2/fcn1/LeakyRelu_1	LeakyRelu&model/decoder2/fcn1/BiasAdd_1:output:0*(
_output_shapes
:         А	2!
model/decoder2/fcn1/LeakyRelu_1╬
+model/decoder1/fcn1/MatMul_1/ReadVariableOpReadVariableOp2model_decoder1_fcn1_matmul_readvariableop_resource*
_output_shapes
:	dА	*
dtype02-
+model/decoder1/fcn1/MatMul_1/ReadVariableOp╘
model/decoder1/fcn1/MatMul_1MatMul$model/tf.__operators__.add/AddV2:z:03model/decoder1/fcn1/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:         А	2
model/decoder1/fcn1/MatMul_1═
,model/decoder1/fcn1/BiasAdd_1/ReadVariableOpReadVariableOp3model_decoder1_fcn1_biasadd_readvariableop_resource*
_output_shapes	
:А	*
dtype02.
,model/decoder1/fcn1/BiasAdd_1/ReadVariableOp┌
model/decoder1/fcn1/BiasAdd_1BiasAdd&model/decoder1/fcn1/MatMul_1:product:04model/decoder1/fcn1/BiasAdd_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:         А	2
model/decoder1/fcn1/BiasAdd_1б
model/decoder1/fcn1/LeakyRelu_1	LeakyRelu&model/decoder1/fcn1/BiasAdd_1:output:0*(
_output_shapes
:         А	2!
model/decoder1/fcn1/LeakyRelu_1й
model/tf.nn.softmax_1/SoftmaxSoftmax#model/tf.math.truediv_1/truediv:z:0*
T0*0
_output_shapes
:                  2
model/tf.nn.softmax_1/Softmaxг
model/tf.nn.softmax/SoftmaxSoftmax!model/tf.math.truediv/truediv:z:0*
T0*0
_output_shapes
:                  2
model/tf.nn.softmax/SoftmaxЕ
model/tf.math.multiply_9/Mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *  HB2 
model/tf.math.multiply_9/Mul/yи
model/tf.math.multiply_9/MulMulinput_5'model/tf.math.multiply_9/Mul/y:output:0*
T0*(
_output_shapes
:         А2
model/tf.math.multiply_9/Mul╦
)model/decoder2/fcn2/MatMul/ReadVariableOpReadVariableOp2model_decoder2_fcn2_matmul_readvariableop_resource* 
_output_shapes
:
А	А*
dtype02+
)model/decoder2/fcn2/MatMul/ReadVariableOp╒
model/decoder2/fcn2/MatMulMatMul+model/decoder2/fcn1/LeakyRelu:activations:01model/decoder2/fcn2/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         А2
model/decoder2/fcn2/MatMul╔
*model/decoder2/fcn2/BiasAdd/ReadVariableOpReadVariableOp3model_decoder2_fcn2_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype02,
*model/decoder2/fcn2/BiasAdd/ReadVariableOp╥
model/decoder2/fcn2/BiasAddBiasAdd$model/decoder2/fcn2/MatMul:product:02model/decoder2/fcn2/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         А2
model/decoder2/fcn2/BiasAddЕ
model/tf.math.multiply_8/Mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *  HB2 
model/tf.math.multiply_8/Mul/yи
model/tf.math.multiply_8/MulMulinput_3'model/tf.math.multiply_8/Mul/y:output:0*
T0*(
_output_shapes
:         А2
model/tf.math.multiply_8/Mul╦
)model/decoder1/fcn2/MatMul/ReadVariableOpReadVariableOp2model_decoder1_fcn2_matmul_readvariableop_resource* 
_output_shapes
:
А	А*
dtype02+
)model/decoder1/fcn2/MatMul/ReadVariableOp╒
model/decoder1/fcn2/MatMulMatMul+model/decoder1/fcn1/LeakyRelu:activations:01model/decoder1/fcn2/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         А2
model/decoder1/fcn2/MatMul╔
*model/decoder1/fcn2/BiasAdd/ReadVariableOpReadVariableOp3model_decoder1_fcn2_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype02,
*model/decoder1/fcn2/BiasAdd/ReadVariableOp╥
model/decoder1/fcn2/BiasAddBiasAdd$model/decoder1/fcn2/MatMul:product:02model/decoder1/fcn2/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         А2
model/decoder1/fcn2/BiasAdd╧
+model/decoder2/fcn2/MatMul_1/ReadVariableOpReadVariableOp2model_decoder2_fcn2_matmul_readvariableop_resource* 
_output_shapes
:
А	А*
dtype02-
+model/decoder2/fcn2/MatMul_1/ReadVariableOp▌
model/decoder2/fcn2/MatMul_1MatMul-model/decoder2/fcn1/LeakyRelu_1:activations:03model/decoder2/fcn2/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:         А2
model/decoder2/fcn2/MatMul_1═
,model/decoder2/fcn2/BiasAdd_1/ReadVariableOpReadVariableOp3model_decoder2_fcn2_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype02.
,model/decoder2/fcn2/BiasAdd_1/ReadVariableOp┌
model/decoder2/fcn2/BiasAdd_1BiasAdd&model/decoder2/fcn2/MatMul_1:product:04model/decoder2/fcn2/BiasAdd_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:         А2
model/decoder2/fcn2/BiasAdd_1╧
+model/decoder1/fcn2/MatMul_1/ReadVariableOpReadVariableOp2model_decoder1_fcn2_matmul_readvariableop_resource* 
_output_shapes
:
А	А*
dtype02-
+model/decoder1/fcn2/MatMul_1/ReadVariableOp▌
model/decoder1/fcn2/MatMul_1MatMul-model/decoder1/fcn1/LeakyRelu_1:activations:03model/decoder1/fcn2/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:         А2
model/decoder1/fcn2/MatMul_1═
,model/decoder1/fcn2/BiasAdd_1/ReadVariableOpReadVariableOp3model_decoder1_fcn2_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype02.
,model/decoder1/fcn2/BiasAdd_1/ReadVariableOp┌
model/decoder1/fcn2/BiasAdd_1BiasAdd&model/decoder1/fcn2/MatMul_1:product:04model/decoder1/fcn2/BiasAdd_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:         А2
model/decoder1/fcn2/BiasAdd_1╪
model/tf.math.truediv_3/truedivRealDiv'model/tf.nn.softmax_1/Softmax:softmax:0%model/tf.nn.softmax/Softmax:softmax:0*
T0*0
_output_shapes
:                  2!
model/tf.math.truediv_3/truediv╪
model/tf.math.truediv_2/truedivRealDiv%model/tf.nn.softmax/Softmax:softmax:0'model/tf.nn.softmax_1/Softmax:softmax:0*
T0*0
_output_shapes
:                  2!
model/tf.math.truediv_2/truediv╛
model/tf.math.subtract_5/SubSub model/tf.math.multiply_9/Mul:z:0$model/decoder2/fcn2/BiasAdd:output:0*
T0*(
_output_shapes
:         А2
model/tf.math.subtract_5/Sub╛
model/tf.math.subtract_4/SubSub model/tf.math.multiply_8/Mul:z:0$model/decoder1/fcn2/BiasAdd:output:0*
T0*(
_output_shapes
:         А2
model/tf.math.subtract_4/SubЕ
model/tf.math.multiply_1/Mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *  HB2 
model/tf.math.multiply_1/Mul/yи
model/tf.math.multiply_1/MulMulinput_2'model/tf.math.multiply_1/Mul/y:output:0*
T0*(
_output_shapes
:         А2
model/tf.math.multiply_1/MulБ
model/tf.math.multiply/Mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *  HB2
model/tf.math.multiply/Mul/yв
model/tf.math.multiply/MulMulinput_1%model/tf.math.multiply/Mul/y:output:0*
T0*(
_output_shapes
:         А2
model/tf.math.multiply/MulЩ
model/tf.math.log_1/LogLog#model/tf.math.truediv_3/truediv:z:0*
T0*0
_output_shapes
:                  2
model/tf.math.log_1/LogХ
model/tf.math.log/LogLog#model/tf.math.truediv_2/truediv:z:0*
T0*0
_output_shapes
:                  2
model/tf.math.log/LogЭ
model/tf.math.square_9/SquareSquare model/tf.math.subtract_5/Sub:z:0*
T0*(
_output_shapes
:         А2
model/tf.math.square_9/SquareЭ
model/tf.math.square_8/SquareSquare model/tf.math.subtract_4/Sub:z:0*
T0*(
_output_shapes
:         А2
model/tf.math.square_8/Square└
model/tf.math.subtract_2/SubSub model/tf.math.multiply_1/Mul:z:0&model/decoder2/fcn2/BiasAdd_1:output:0*
T0*(
_output_shapes
:         А2
model/tf.math.subtract_2/Sub╛
model/tf.math.subtract_1/SubSubmodel/tf.math.multiply/Mul:z:0&model/decoder1/fcn2/BiasAdd_1:output:0*
T0*(
_output_shapes
:         А2
model/tf.math.subtract_1/Sub─
model/tf.math.multiply_5/MulMul'model/tf.nn.softmax_1/Softmax:softmax:0model/tf.math.log_1/Log:y:0*
T0*0
_output_shapes
:                  2
model/tf.math.multiply_5/Mul└
model/tf.math.multiply_4/MulMul%model/tf.nn.softmax/Softmax:softmax:0model/tf.math.log/Log:y:0*
T0*0
_output_shapes
:                  2
model/tf.math.multiply_4/MulЧ
!model/tf.math.reduce_mean_3/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2#
!model/tf.math.reduce_mean_3/Const╝
 model/tf.math.reduce_mean_3/MeanMean!model/tf.math.square_8/Square:y:0*model/tf.math.reduce_mean_3/Const:output:0*
T0*
_output_shapes
: 2"
 model/tf.math.reduce_mean_3/MeanЧ
!model/tf.math.reduce_mean_4/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2#
!model/tf.math.reduce_mean_4/Const╝
 model/tf.math.reduce_mean_4/MeanMean!model/tf.math.square_9/Square:y:0*model/tf.math.reduce_mean_4/Const:output:0*
T0*
_output_shapes
: 2"
 model/tf.math.reduce_mean_4/MeanЭ
model/tf.math.square_1/SquareSquare model/tf.math.subtract_2/Sub:z:0*
T0*(
_output_shapes
:         А2
model/tf.math.square_1/SquareЩ
model/tf.math.square/SquareSquare model/tf.math.subtract_1/Sub:z:0*
T0*(
_output_shapes
:         А2
model/tf.math.square/Square╔
)model/encoder2/fcn3/MatMul/ReadVariableOpReadVariableOp2model_encoder2_fcn3_matmul_readvariableop_resource*
_output_shapes

:`P*
dtype02+
)model/encoder2/fcn3/MatMul/ReadVariableOp┘
model/encoder2/fcn3/MatMulMatMul0model/encoder2/fcn2/mean/LeakyRelu:activations:01model/encoder2/fcn3/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         P2
model/encoder2/fcn3/MatMul╚
*model/encoder2/fcn3/BiasAdd/ReadVariableOpReadVariableOp3model_encoder2_fcn3_biasadd_readvariableop_resource*
_output_shapes
:P*
dtype02,
*model/encoder2/fcn3/BiasAdd/ReadVariableOp╤
model/encoder2/fcn3/BiasAddBiasAdd$model/encoder2/fcn3/MatMul:product:02model/encoder2/fcn3/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         P2
model/encoder2/fcn3/BiasAdd╔
)model/encoder1/fcn3/MatMul/ReadVariableOpReadVariableOp2model_encoder1_fcn3_matmul_readvariableop_resource*
_output_shapes

:dP*
dtype02+
)model/encoder1/fcn3/MatMul/ReadVariableOp█
model/encoder1/fcn3/MatMulMatMul2model/encoder1/fcn2/mean/LeakyRelu_1:activations:01model/encoder1/fcn3/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         P2
model/encoder1/fcn3/MatMul╚
*model/encoder1/fcn3/BiasAdd/ReadVariableOpReadVariableOp3model_encoder1_fcn3_biasadd_readvariableop_resource*
_output_shapes
:P*
dtype02,
*model/encoder1/fcn3/BiasAdd/ReadVariableOp╤
model/encoder1/fcn3/BiasAddBiasAdd$model/encoder1/fcn3/MatMul:product:02model/encoder1/fcn3/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         P2
model/encoder1/fcn3/BiasAddп
0model/tf.math.reduce_sum_4/Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
valueB :
         22
0model/tf.math.reduce_sum_4/Sum/reduction_indices╥
model/tf.math.reduce_sum_4/SumSum model/tf.math.multiply_4/Mul:z:09model/tf.math.reduce_sum_4/Sum/reduction_indices:output:0*
T0*#
_output_shapes
:         2 
model/tf.math.reduce_sum_4/Sumп
0model/tf.math.reduce_sum_5/Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
valueB :
         22
0model/tf.math.reduce_sum_5/Sum/reduction_indices╥
model/tf.math.reduce_sum_5/SumSum model/tf.math.multiply_5/Mul:z:09model/tf.math.reduce_sum_5/Sum/reduction_indices:output:0*
T0*#
_output_shapes
:         2 
model/tf.math.reduce_sum_5/Sum╩
#model/tf.__operators__.add_16/AddV2AddV2)model/tf.math.reduce_mean_3/Mean:output:0)model/tf.math.reduce_mean_4/Mean:output:0*
T0*
_output_shapes
: 2%
#model/tf.__operators__.add_16/AddV2У
model/tf.math.reduce_mean/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2!
model/tf.math.reduce_mean/Const┤
model/tf.math.reduce_mean/MeanMeanmodel/tf.math.square/Square:y:0(model/tf.math.reduce_mean/Const:output:0*
T0*
_output_shapes
: 2 
model/tf.math.reduce_mean/MeanЧ
!model/tf.math.reduce_mean_1/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2#
!model/tf.math.reduce_mean_1/Const╝
 model/tf.math.reduce_mean_1/MeanMean!model/tf.math.square_1/Square:y:0*model/tf.math.reduce_mean_1/Const:output:0*
T0*
_output_shapes
: 2"
 model/tf.math.reduce_mean_1/Meanа
model/tf.math.square_7/SquareSquare$model/encoder2/fcn3/BiasAdd:output:0*
T0*'
_output_shapes
:         P2
model/tf.math.square_7/Squareа
model/tf.math.square_6/SquareSquare$model/encoder1/fcn3/BiasAdd:output:0*
T0*'
_output_shapes
:         P2
model/tf.math.square_6/Square╤
"model/tf.__operators__.add_1/AddV2AddV2'model/tf.math.reduce_sum_4/Sum:output:0'model/tf.math.reduce_sum_5/Sum:output:0*
T0*#
_output_shapes
:         2$
"model/tf.__operators__.add_1/AddV2Я
model/tf.math.multiply_22/MulMulmodel_40049411'model/tf.__operators__.add_16/AddV2:z:0*
T0*
_output_shapes
: 2
model/tf.math.multiply_22/Mul╚
#model/tf.__operators__.add_15/AddV2AddV2'model/tf.math.reduce_mean/Mean:output:0)model/tf.math.reduce_mean_1/Mean:output:0*
T0*
_output_shapes
: 2%
#model/tf.__operators__.add_15/AddV2п
0model/tf.math.reduce_sum_8/Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
valueB :
         22
0model/tf.math.reduce_sum_8/Sum/reduction_indices╙
model/tf.math.reduce_sum_8/SumSum!model/tf.math.square_7/Square:y:09model/tf.math.reduce_sum_8/Sum/reduction_indices:output:0*
T0*#
_output_shapes
:         2 
model/tf.math.reduce_sum_8/Sumп
0model/tf.math.reduce_sum_7/Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
valueB :
         22
0model/tf.math.reduce_sum_7/Sum/reduction_indices╙
model/tf.math.reduce_sum_7/SumSum!model/tf.math.square_6/Square:y:09model/tf.math.reduce_sum_7/Sum/reduction_indices:output:0*
T0*#
_output_shapes
:         2 
model/tf.math.reduce_sum_7/SumР
!model/tf.math.reduce_mean_2/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2#
!model/tf.math.reduce_mean_2/Const┴
 model/tf.math.reduce_mean_2/MeanMean&model/tf.__operators__.add_1/AddV2:z:0*model/tf.math.reduce_mean_2/Const:output:0*
T0*
_output_shapes
: 2"
 model/tf.math.reduce_mean_2/MeanН
"model/tf.math.truediv_21/truediv/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @2$
"model/tf.math.truediv_21/truediv/y╞
 model/tf.math.truediv_21/truedivRealDiv'model/tf.__operators__.add_15/AddV2:z:0+model/tf.math.truediv_21/truediv/y:output:0*
T0*
_output_shapes
: 2"
 model/tf.math.truediv_21/truedivН
"model/tf.math.truediv_22/truediv/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @2$
"model/tf.math.truediv_22/truediv/y└
 model/tf.math.truediv_22/truedivRealDiv!model/tf.math.multiply_22/Mul:z:0+model/tf.math.truediv_22/truediv/y:output:0*
T0*
_output_shapes
: 2"
 model/tf.math.truediv_22/truedivХ
model/tf.math.sqrt_4/SqrtSqrt'model/tf.math.reduce_sum_7/Sum:output:0*
T0*#
_output_shapes
:         2
model/tf.math.sqrt_4/SqrtХ
model/tf.math.sqrt_5/SqrtSqrt'model/tf.math.reduce_sum_8/Sum:output:0*
T0*#
_output_shapes
:         2
model/tf.math.sqrt_5/Sqrt┴
model/tf.math.multiply_6/MulMul$model/encoder1/fcn3/BiasAdd:output:0$model/encoder2/fcn3/BiasAdd:output:0*
T0*'
_output_shapes
:         P2
model/tf.math.multiply_6/Mul└
#model/tf.__operators__.add_17/AddV2AddV2$model/tf.math.truediv_21/truediv:z:0$model/tf.math.truediv_22/truediv:z:0*
T0*
_output_shapes
: 2%
#model/tf.__operators__.add_17/AddV2З
model/tf.math.multiply_23/Mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2!
model/tf.math.multiply_23/Mul/y╗
model/tf.math.multiply_23/MulMul)model/tf.math.reduce_mean_2/Mean:output:0(model/tf.math.multiply_23/Mul/y:output:0*
T0*
_output_shapes
: 2
model/tf.math.multiply_23/Mulп
0model/tf.math.reduce_sum_6/Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
valueB :
         22
0model/tf.math.reduce_sum_6/Sum/reduction_indices╥
model/tf.math.reduce_sum_6/SumSum model/tf.math.multiply_6/Mul:z:09model/tf.math.reduce_sum_6/Sum/reduction_indices:output:0*
T0*#
_output_shapes
:         2 
model/tf.math.reduce_sum_6/Sumп
model/tf.math.multiply_7/MulMulmodel/tf.math.sqrt_4/Sqrt:y:0model/tf.math.sqrt_5/Sqrt:y:0*
T0*#
_output_shapes
:         2
model/tf.math.multiply_7/Mul└
#model/tf.__operators__.add_18/AddV2AddV2'model/tf.__operators__.add_17/AddV2:z:0!model/tf.math.multiply_23/Mul:z:0*
T0*
_output_shapes
: 2%
#model/tf.__operators__.add_18/AddV2╞
model/tf.math.truediv_4/truedivRealDiv'model/tf.math.reduce_sum_6/Sum:output:0 model/tf.math.multiply_7/Mul:z:0*
T0*#
_output_shapes
:         2!
model/tf.math.truediv_4/truedivЪ
IdentityIdentity$model/encoder1/fcn3/BiasAdd:output:0+^model/decoder1/fcn1/BiasAdd/ReadVariableOp-^model/decoder1/fcn1/BiasAdd_1/ReadVariableOp*^model/decoder1/fcn1/MatMul/ReadVariableOp,^model/decoder1/fcn1/MatMul_1/ReadVariableOp+^model/decoder1/fcn2/BiasAdd/ReadVariableOp-^model/decoder1/fcn2/BiasAdd_1/ReadVariableOp*^model/decoder1/fcn2/MatMul/ReadVariableOp,^model/decoder1/fcn2/MatMul_1/ReadVariableOp1^model/decoder1/layer_gene/BiasAdd/ReadVariableOp3^model/decoder1/layer_gene/BiasAdd_1/ReadVariableOp0^model/decoder1/layer_gene/MatMul/ReadVariableOp2^model/decoder1/layer_gene/MatMul_1/ReadVariableOp+^model/decoder2/fcn1/BiasAdd/ReadVariableOp-^model/decoder2/fcn1/BiasAdd_1/ReadVariableOp*^model/decoder2/fcn1/MatMul/ReadVariableOp,^model/decoder2/fcn1/MatMul_1/ReadVariableOp+^model/decoder2/fcn2/BiasAdd/ReadVariableOp-^model/decoder2/fcn2/BiasAdd_1/ReadVariableOp*^model/decoder2/fcn2/MatMul/ReadVariableOp,^model/decoder2/fcn2/MatMul_1/ReadVariableOp+^model/encoder1/fcn1/BiasAdd/ReadVariableOp-^model/encoder1/fcn1/BiasAdd_1/ReadVariableOp*^model/encoder1/fcn1/MatMul/ReadVariableOp,^model/encoder1/fcn1/MatMul_1/ReadVariableOp0^model/encoder1/fcn2/mean/BiasAdd/ReadVariableOp2^model/encoder1/fcn2/mean/BiasAdd_1/ReadVariableOp/^model/encoder1/fcn2/mean/MatMul/ReadVariableOp1^model/encoder1/fcn2/mean/MatMul_1/ReadVariableOp+^model/encoder1/fcn3/BiasAdd/ReadVariableOp*^model/encoder1/fcn3/MatMul/ReadVariableOp0^model/encoder1/fcn3_gene/BiasAdd/ReadVariableOp2^model/encoder1/fcn3_gene/BiasAdd_1/ReadVariableOp/^model/encoder1/fcn3_gene/MatMul/ReadVariableOp1^model/encoder1/fcn3_gene/MatMul_1/ReadVariableOp+^model/encoder2/fcn1/BiasAdd/ReadVariableOp-^model/encoder2/fcn1/BiasAdd_1/ReadVariableOp*^model/encoder2/fcn1/MatMul/ReadVariableOp,^model/encoder2/fcn1/MatMul_1/ReadVariableOp0^model/encoder2/fcn2/mean/BiasAdd/ReadVariableOp2^model/encoder2/fcn2/mean/BiasAdd_1/ReadVariableOp/^model/encoder2/fcn2/mean/MatMul/ReadVariableOp1^model/encoder2/fcn2/mean/MatMul_1/ReadVariableOp+^model/encoder2/fcn3/BiasAdd/ReadVariableOp*^model/encoder2/fcn3/MatMul/ReadVariableOp*
T0*'
_output_shapes
:         P2

IdentityЮ

Identity_1Identity$model/encoder2/fcn3/BiasAdd:output:0+^model/decoder1/fcn1/BiasAdd/ReadVariableOp-^model/decoder1/fcn1/BiasAdd_1/ReadVariableOp*^model/decoder1/fcn1/MatMul/ReadVariableOp,^model/decoder1/fcn1/MatMul_1/ReadVariableOp+^model/decoder1/fcn2/BiasAdd/ReadVariableOp-^model/decoder1/fcn2/BiasAdd_1/ReadVariableOp*^model/decoder1/fcn2/MatMul/ReadVariableOp,^model/decoder1/fcn2/MatMul_1/ReadVariableOp1^model/decoder1/layer_gene/BiasAdd/ReadVariableOp3^model/decoder1/layer_gene/BiasAdd_1/ReadVariableOp0^model/decoder1/layer_gene/MatMul/ReadVariableOp2^model/decoder1/layer_gene/MatMul_1/ReadVariableOp+^model/decoder2/fcn1/BiasAdd/ReadVariableOp-^model/decoder2/fcn1/BiasAdd_1/ReadVariableOp*^model/decoder2/fcn1/MatMul/ReadVariableOp,^model/decoder2/fcn1/MatMul_1/ReadVariableOp+^model/decoder2/fcn2/BiasAdd/ReadVariableOp-^model/decoder2/fcn2/BiasAdd_1/ReadVariableOp*^model/decoder2/fcn2/MatMul/ReadVariableOp,^model/decoder2/fcn2/MatMul_1/ReadVariableOp+^model/encoder1/fcn1/BiasAdd/ReadVariableOp-^model/encoder1/fcn1/BiasAdd_1/ReadVariableOp*^model/encoder1/fcn1/MatMul/ReadVariableOp,^model/encoder1/fcn1/MatMul_1/ReadVariableOp0^model/encoder1/fcn2/mean/BiasAdd/ReadVariableOp2^model/encoder1/fcn2/mean/BiasAdd_1/ReadVariableOp/^model/encoder1/fcn2/mean/MatMul/ReadVariableOp1^model/encoder1/fcn2/mean/MatMul_1/ReadVariableOp+^model/encoder1/fcn3/BiasAdd/ReadVariableOp*^model/encoder1/fcn3/MatMul/ReadVariableOp0^model/encoder1/fcn3_gene/BiasAdd/ReadVariableOp2^model/encoder1/fcn3_gene/BiasAdd_1/ReadVariableOp/^model/encoder1/fcn3_gene/MatMul/ReadVariableOp1^model/encoder1/fcn3_gene/MatMul_1/ReadVariableOp+^model/encoder2/fcn1/BiasAdd/ReadVariableOp-^model/encoder2/fcn1/BiasAdd_1/ReadVariableOp*^model/encoder2/fcn1/MatMul/ReadVariableOp,^model/encoder2/fcn1/MatMul_1/ReadVariableOp0^model/encoder2/fcn2/mean/BiasAdd/ReadVariableOp2^model/encoder2/fcn2/mean/BiasAdd_1/ReadVariableOp/^model/encoder2/fcn2/mean/MatMul/ReadVariableOp1^model/encoder2/fcn2/mean/MatMul_1/ReadVariableOp+^model/encoder2/fcn3/BiasAdd/ReadVariableOp*^model/encoder2/fcn3/MatMul/ReadVariableOp*
T0*'
_output_shapes
:         P2

Identity_1Р

Identity_2Identity'model/tf.__operators__.add_18/AddV2:z:0+^model/decoder1/fcn1/BiasAdd/ReadVariableOp-^model/decoder1/fcn1/BiasAdd_1/ReadVariableOp*^model/decoder1/fcn1/MatMul/ReadVariableOp,^model/decoder1/fcn1/MatMul_1/ReadVariableOp+^model/decoder1/fcn2/BiasAdd/ReadVariableOp-^model/decoder1/fcn2/BiasAdd_1/ReadVariableOp*^model/decoder1/fcn2/MatMul/ReadVariableOp,^model/decoder1/fcn2/MatMul_1/ReadVariableOp1^model/decoder1/layer_gene/BiasAdd/ReadVariableOp3^model/decoder1/layer_gene/BiasAdd_1/ReadVariableOp0^model/decoder1/layer_gene/MatMul/ReadVariableOp2^model/decoder1/layer_gene/MatMul_1/ReadVariableOp+^model/decoder2/fcn1/BiasAdd/ReadVariableOp-^model/decoder2/fcn1/BiasAdd_1/ReadVariableOp*^model/decoder2/fcn1/MatMul/ReadVariableOp,^model/decoder2/fcn1/MatMul_1/ReadVariableOp+^model/decoder2/fcn2/BiasAdd/ReadVariableOp-^model/decoder2/fcn2/BiasAdd_1/ReadVariableOp*^model/decoder2/fcn2/MatMul/ReadVariableOp,^model/decoder2/fcn2/MatMul_1/ReadVariableOp+^model/encoder1/fcn1/BiasAdd/ReadVariableOp-^model/encoder1/fcn1/BiasAdd_1/ReadVariableOp*^model/encoder1/fcn1/MatMul/ReadVariableOp,^model/encoder1/fcn1/MatMul_1/ReadVariableOp0^model/encoder1/fcn2/mean/BiasAdd/ReadVariableOp2^model/encoder1/fcn2/mean/BiasAdd_1/ReadVariableOp/^model/encoder1/fcn2/mean/MatMul/ReadVariableOp1^model/encoder1/fcn2/mean/MatMul_1/ReadVariableOp+^model/encoder1/fcn3/BiasAdd/ReadVariableOp*^model/encoder1/fcn3/MatMul/ReadVariableOp0^model/encoder1/fcn3_gene/BiasAdd/ReadVariableOp2^model/encoder1/fcn3_gene/BiasAdd_1/ReadVariableOp/^model/encoder1/fcn3_gene/MatMul/ReadVariableOp1^model/encoder1/fcn3_gene/MatMul_1/ReadVariableOp+^model/encoder2/fcn1/BiasAdd/ReadVariableOp-^model/encoder2/fcn1/BiasAdd_1/ReadVariableOp*^model/encoder2/fcn1/MatMul/ReadVariableOp,^model/encoder2/fcn1/MatMul_1/ReadVariableOp0^model/encoder2/fcn2/mean/BiasAdd/ReadVariableOp2^model/encoder2/fcn2/mean/BiasAdd_1/ReadVariableOp/^model/encoder2/fcn2/mean/MatMul/ReadVariableOp1^model/encoder2/fcn2/mean/MatMul_1/ReadVariableOp+^model/encoder2/fcn3/BiasAdd/ReadVariableOp*^model/encoder2/fcn3/MatMul/ReadVariableOp*
T0*
_output_shapes
: 2

Identity_2Щ

Identity_3Identity#model/tf.math.truediv_4/truediv:z:0+^model/decoder1/fcn1/BiasAdd/ReadVariableOp-^model/decoder1/fcn1/BiasAdd_1/ReadVariableOp*^model/decoder1/fcn1/MatMul/ReadVariableOp,^model/decoder1/fcn1/MatMul_1/ReadVariableOp+^model/decoder1/fcn2/BiasAdd/ReadVariableOp-^model/decoder1/fcn2/BiasAdd_1/ReadVariableOp*^model/decoder1/fcn2/MatMul/ReadVariableOp,^model/decoder1/fcn2/MatMul_1/ReadVariableOp1^model/decoder1/layer_gene/BiasAdd/ReadVariableOp3^model/decoder1/layer_gene/BiasAdd_1/ReadVariableOp0^model/decoder1/layer_gene/MatMul/ReadVariableOp2^model/decoder1/layer_gene/MatMul_1/ReadVariableOp+^model/decoder2/fcn1/BiasAdd/ReadVariableOp-^model/decoder2/fcn1/BiasAdd_1/ReadVariableOp*^model/decoder2/fcn1/MatMul/ReadVariableOp,^model/decoder2/fcn1/MatMul_1/ReadVariableOp+^model/decoder2/fcn2/BiasAdd/ReadVariableOp-^model/decoder2/fcn2/BiasAdd_1/ReadVariableOp*^model/decoder2/fcn2/MatMul/ReadVariableOp,^model/decoder2/fcn2/MatMul_1/ReadVariableOp+^model/encoder1/fcn1/BiasAdd/ReadVariableOp-^model/encoder1/fcn1/BiasAdd_1/ReadVariableOp*^model/encoder1/fcn1/MatMul/ReadVariableOp,^model/encoder1/fcn1/MatMul_1/ReadVariableOp0^model/encoder1/fcn2/mean/BiasAdd/ReadVariableOp2^model/encoder1/fcn2/mean/BiasAdd_1/ReadVariableOp/^model/encoder1/fcn2/mean/MatMul/ReadVariableOp1^model/encoder1/fcn2/mean/MatMul_1/ReadVariableOp+^model/encoder1/fcn3/BiasAdd/ReadVariableOp*^model/encoder1/fcn3/MatMul/ReadVariableOp0^model/encoder1/fcn3_gene/BiasAdd/ReadVariableOp2^model/encoder1/fcn3_gene/BiasAdd_1/ReadVariableOp/^model/encoder1/fcn3_gene/MatMul/ReadVariableOp1^model/encoder1/fcn3_gene/MatMul_1/ReadVariableOp+^model/encoder2/fcn1/BiasAdd/ReadVariableOp-^model/encoder2/fcn1/BiasAdd_1/ReadVariableOp*^model/encoder2/fcn1/MatMul/ReadVariableOp,^model/encoder2/fcn1/MatMul_1/ReadVariableOp0^model/encoder2/fcn2/mean/BiasAdd/ReadVariableOp2^model/encoder2/fcn2/mean/BiasAdd_1/ReadVariableOp/^model/encoder2/fcn2/mean/MatMul/ReadVariableOp1^model/encoder2/fcn2/mean/MatMul_1/ReadVariableOp+^model/encoder2/fcn3/BiasAdd/ReadVariableOp*^model/encoder2/fcn3/MatMul/ReadVariableOp*
T0*#
_output_shapes
:         2

Identity_3"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0*(
_construction_contextkEagerRuntime*┐
_input_shapesн
к:         А:         А:         А:         А:         А:         А: : : : : : : : : : : : : : : : : : : : : : : : : 2X
*model/decoder1/fcn1/BiasAdd/ReadVariableOp*model/decoder1/fcn1/BiasAdd/ReadVariableOp2\
,model/decoder1/fcn1/BiasAdd_1/ReadVariableOp,model/decoder1/fcn1/BiasAdd_1/ReadVariableOp2V
)model/decoder1/fcn1/MatMul/ReadVariableOp)model/decoder1/fcn1/MatMul/ReadVariableOp2Z
+model/decoder1/fcn1/MatMul_1/ReadVariableOp+model/decoder1/fcn1/MatMul_1/ReadVariableOp2X
*model/decoder1/fcn2/BiasAdd/ReadVariableOp*model/decoder1/fcn2/BiasAdd/ReadVariableOp2\
,model/decoder1/fcn2/BiasAdd_1/ReadVariableOp,model/decoder1/fcn2/BiasAdd_1/ReadVariableOp2V
)model/decoder1/fcn2/MatMul/ReadVariableOp)model/decoder1/fcn2/MatMul/ReadVariableOp2Z
+model/decoder1/fcn2/MatMul_1/ReadVariableOp+model/decoder1/fcn2/MatMul_1/ReadVariableOp2d
0model/decoder1/layer_gene/BiasAdd/ReadVariableOp0model/decoder1/layer_gene/BiasAdd/ReadVariableOp2h
2model/decoder1/layer_gene/BiasAdd_1/ReadVariableOp2model/decoder1/layer_gene/BiasAdd_1/ReadVariableOp2b
/model/decoder1/layer_gene/MatMul/ReadVariableOp/model/decoder1/layer_gene/MatMul/ReadVariableOp2f
1model/decoder1/layer_gene/MatMul_1/ReadVariableOp1model/decoder1/layer_gene/MatMul_1/ReadVariableOp2X
*model/decoder2/fcn1/BiasAdd/ReadVariableOp*model/decoder2/fcn1/BiasAdd/ReadVariableOp2\
,model/decoder2/fcn1/BiasAdd_1/ReadVariableOp,model/decoder2/fcn1/BiasAdd_1/ReadVariableOp2V
)model/decoder2/fcn1/MatMul/ReadVariableOp)model/decoder2/fcn1/MatMul/ReadVariableOp2Z
+model/decoder2/fcn1/MatMul_1/ReadVariableOp+model/decoder2/fcn1/MatMul_1/ReadVariableOp2X
*model/decoder2/fcn2/BiasAdd/ReadVariableOp*model/decoder2/fcn2/BiasAdd/ReadVariableOp2\
,model/decoder2/fcn2/BiasAdd_1/ReadVariableOp,model/decoder2/fcn2/BiasAdd_1/ReadVariableOp2V
)model/decoder2/fcn2/MatMul/ReadVariableOp)model/decoder2/fcn2/MatMul/ReadVariableOp2Z
+model/decoder2/fcn2/MatMul_1/ReadVariableOp+model/decoder2/fcn2/MatMul_1/ReadVariableOp2X
*model/encoder1/fcn1/BiasAdd/ReadVariableOp*model/encoder1/fcn1/BiasAdd/ReadVariableOp2\
,model/encoder1/fcn1/BiasAdd_1/ReadVariableOp,model/encoder1/fcn1/BiasAdd_1/ReadVariableOp2V
)model/encoder1/fcn1/MatMul/ReadVariableOp)model/encoder1/fcn1/MatMul/ReadVariableOp2Z
+model/encoder1/fcn1/MatMul_1/ReadVariableOp+model/encoder1/fcn1/MatMul_1/ReadVariableOp2b
/model/encoder1/fcn2/mean/BiasAdd/ReadVariableOp/model/encoder1/fcn2/mean/BiasAdd/ReadVariableOp2f
1model/encoder1/fcn2/mean/BiasAdd_1/ReadVariableOp1model/encoder1/fcn2/mean/BiasAdd_1/ReadVariableOp2`
.model/encoder1/fcn2/mean/MatMul/ReadVariableOp.model/encoder1/fcn2/mean/MatMul/ReadVariableOp2d
0model/encoder1/fcn2/mean/MatMul_1/ReadVariableOp0model/encoder1/fcn2/mean/MatMul_1/ReadVariableOp2X
*model/encoder1/fcn3/BiasAdd/ReadVariableOp*model/encoder1/fcn3/BiasAdd/ReadVariableOp2V
)model/encoder1/fcn3/MatMul/ReadVariableOp)model/encoder1/fcn3/MatMul/ReadVariableOp2b
/model/encoder1/fcn3_gene/BiasAdd/ReadVariableOp/model/encoder1/fcn3_gene/BiasAdd/ReadVariableOp2f
1model/encoder1/fcn3_gene/BiasAdd_1/ReadVariableOp1model/encoder1/fcn3_gene/BiasAdd_1/ReadVariableOp2`
.model/encoder1/fcn3_gene/MatMul/ReadVariableOp.model/encoder1/fcn3_gene/MatMul/ReadVariableOp2d
0model/encoder1/fcn3_gene/MatMul_1/ReadVariableOp0model/encoder1/fcn3_gene/MatMul_1/ReadVariableOp2X
*model/encoder2/fcn1/BiasAdd/ReadVariableOp*model/encoder2/fcn1/BiasAdd/ReadVariableOp2\
,model/encoder2/fcn1/BiasAdd_1/ReadVariableOp,model/encoder2/fcn1/BiasAdd_1/ReadVariableOp2V
)model/encoder2/fcn1/MatMul/ReadVariableOp)model/encoder2/fcn1/MatMul/ReadVariableOp2Z
+model/encoder2/fcn1/MatMul_1/ReadVariableOp+model/encoder2/fcn1/MatMul_1/ReadVariableOp2b
/model/encoder2/fcn2/mean/BiasAdd/ReadVariableOp/model/encoder2/fcn2/mean/BiasAdd/ReadVariableOp2f
1model/encoder2/fcn2/mean/BiasAdd_1/ReadVariableOp1model/encoder2/fcn2/mean/BiasAdd_1/ReadVariableOp2`
.model/encoder2/fcn2/mean/MatMul/ReadVariableOp.model/encoder2/fcn2/mean/MatMul/ReadVariableOp2d
0model/encoder2/fcn2/mean/MatMul_1/ReadVariableOp0model/encoder2/fcn2/mean/MatMul_1/ReadVariableOp2X
*model/encoder2/fcn3/BiasAdd/ReadVariableOp*model/encoder2/fcn3/BiasAdd/ReadVariableOp2V
)model/encoder2/fcn3/MatMul/ReadVariableOp)model/encoder2/fcn3/MatMul/ReadVariableOp:Q M
(
_output_shapes
:         А
!
_user_specified_name	input_1:QM
(
_output_shapes
:         А
!
_user_specified_name	input_2:QM
(
_output_shapes
:         А
!
_user_specified_name	input_3:QM
(
_output_shapes
:         А
!
_user_specified_name	input_4:QM
(
_output_shapes
:         А
!
_user_specified_name	input_5:QM
(
_output_shapes
:         А
!
_user_specified_name	input_6:

_output_shapes
: 
╫	
№
K__inference_encoder2/fcn3_layer_call_and_return_conditional_losses_40049718

inputs0
matmul_readvariableop_resource:`P-
biasadd_readvariableop_resource:P
identityИвBiasAdd/ReadVariableOpвMatMul/ReadVariableOpН
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:`P*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         P2
MatMulМ
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:P*
dtype02
BiasAdd/ReadVariableOpБ
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         P2	
BiasAddХ
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:         P2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:         `: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:         `
 
_user_specified_nameinputs
├

Б
P__inference_encoder1/fcn2/mean_layer_call_and_return_conditional_losses_40049555

inputs0
matmul_readvariableop_resource:dd-
biasadd_readvariableop_resource:d
identityИвBiasAdd/ReadVariableOpвMatMul/ReadVariableOpН
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:dd*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         d2
MatMulМ
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:d*
dtype02
BiasAdd/ReadVariableOpБ
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         d2	
BiasAdd^
	LeakyRelu	LeakyReluBiasAdd:output:0*'
_output_shapes
:         d2
	LeakyReluЬ
IdentityIdentityLeakyRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:         d2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:         d: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:         d
 
_user_specified_nameinputs
╢
в
5__inference_encoder2/fcn2/mean_layer_call_fn_40051474

inputs
unknown:x`
	unknown_0:`
identityИвStatefulPartitionedCallГ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         `*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *Y
fTRR
P__inference_encoder2/fcn2/mean_layer_call_and_return_conditional_losses_400494852
StatefulPartitionedCallО
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:         `2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:         x: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:         x
 
_user_specified_nameinputs
Ц
Й
(__inference_model_layer_call_fn_40051329
inputs_0
inputs_1
inputs_2
inputs_3
inputs_4
inputs_5
unknown:	Аx
	unknown_0:x
	unknown_1:x`
	unknown_2:`
	unknown_3:	Аd
	unknown_4:d
	unknown_5:	Аd
	unknown_6:d
	unknown_7:dd
	unknown_8:d
	unknown_9:dd

unknown_10:d

unknown_11:	`А	

unknown_12:	А	

unknown_13:	dА	

unknown_14:	А	

unknown_15:
А	А

unknown_16:	А

unknown_17:
А	А

unknown_18:	А

unknown_19:`P

unknown_20:P

unknown_21:dP

unknown_22:P

unknown_23
identity

identity_1

identity_2

identity_3ИвStatefulPartitionedCallО
StatefulPartitionedCallStatefulPartitionedCallinputs_0inputs_1inputs_2inputs_3inputs_4inputs_5unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23**
Tin#
!2*
Tout
2*
_collective_manager_ids
 *K
_output_shapes9
7:         : :         P:         P*:
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8В *L
fGRE
C__inference_model_layer_call_and_return_conditional_losses_400497812
StatefulPartitionedCallК
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*#
_output_shapes
:         2

IdentityБ

Identity_1Identity StatefulPartitionedCall:output:1^StatefulPartitionedCall*
T0*
_output_shapes
: 2

Identity_1Т

Identity_2Identity StatefulPartitionedCall:output:2^StatefulPartitionedCall*
T0*'
_output_shapes
:         P2

Identity_2Т

Identity_3Identity StatefulPartitionedCall:output:3^StatefulPartitionedCall*
T0*'
_output_shapes
:         P2

Identity_3"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0*(
_construction_contextkEagerRuntime*┐
_input_shapesн
к:         А:         А:         А:         А:         А:         А: : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:R N
(
_output_shapes
:         А
"
_user_specified_name
inputs/0:RN
(
_output_shapes
:         А
"
_user_specified_name
inputs/1:RN
(
_output_shapes
:         А
"
_user_specified_name
inputs/2:RN
(
_output_shapes
:         А
"
_user_specified_name
inputs/3:RN
(
_output_shapes
:         А
"
_user_specified_name
inputs/4:RN
(
_output_shapes
:         А
"
_user_specified_name
inputs/5:

_output_shapes
: 
░
Я
0__inference_decoder2/fcn1_layer_call_fn_40051554

inputs
unknown:	`А	
	unknown_0:	А	
identityИвStatefulPartitionedCall 
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         А	*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *T
fORM
K__inference_decoder2/fcn1_layer_call_and_return_conditional_losses_400496132
StatefulPartitionedCallП
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:         А	2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:         `: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:         `
 
_user_specified_nameinputs
─9
·

!__inference__traced_save_40051734
file_prefix3
/savev2_encoder2_fcn1_kernel_read_readvariableop1
-savev2_encoder2_fcn1_bias_read_readvariableop3
/savev2_encoder1_fcn1_kernel_read_readvariableop1
-savev2_encoder1_fcn1_bias_read_readvariableop8
4savev2_encoder1_fcn3_gene_kernel_read_readvariableop6
2savev2_encoder1_fcn3_gene_bias_read_readvariableop8
4savev2_encoder2_fcn2_mean_kernel_read_readvariableop6
2savev2_encoder2_fcn2_mean_bias_read_readvariableop8
4savev2_encoder1_fcn2_mean_kernel_read_readvariableop6
2savev2_encoder1_fcn2_mean_bias_read_readvariableop9
5savev2_decoder1_layer_gene_kernel_read_readvariableop7
3savev2_decoder1_layer_gene_bias_read_readvariableop3
/savev2_decoder1_fcn1_kernel_read_readvariableop1
-savev2_decoder1_fcn1_bias_read_readvariableop3
/savev2_decoder2_fcn1_kernel_read_readvariableop1
-savev2_decoder2_fcn1_bias_read_readvariableop3
/savev2_decoder1_fcn2_kernel_read_readvariableop1
-savev2_decoder1_fcn2_bias_read_readvariableop3
/savev2_decoder2_fcn2_kernel_read_readvariableop1
-savev2_decoder2_fcn2_bias_read_readvariableop3
/savev2_encoder1_fcn3_kernel_read_readvariableop1
-savev2_encoder1_fcn3_bias_read_readvariableop3
/savev2_encoder2_fcn3_kernel_read_readvariableop1
-savev2_encoder2_fcn3_bias_read_readvariableop
savev2_const_1

identity_1ИвMergeV2CheckpointsП
StaticRegexFullMatchStaticRegexFullMatchfile_prefix"/device:CPU:**
_output_shapes
: *
pattern
^s3://.*2
StaticRegexFullMatchc
ConstConst"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B.part2
Constl
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B
_temp/part2	
Const_1Л
SelectSelectStaticRegexFullMatch:output:0Const:output:0Const_1:output:0"/device:CPU:**
T0*
_output_shapes
: 2
Selectt

StringJoin
StringJoinfile_prefixSelect:output:0"/device:CPU:**
N*
_output_shapes
: 2

StringJoinZ

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :2

num_shards
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : 2
ShardedFilename/shardж
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: 2
ShardedFilename═
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*▀

value╒
B╥
B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-7/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-7/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-8/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-8/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-9/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-9/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-10/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-10/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-11/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-11/bias/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
SaveV2/tensor_names║
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*E
value<B:B B B B B B B B B B B B B B B B B B B B B B B B B 2
SaveV2/shape_and_slices■

SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0/savev2_encoder2_fcn1_kernel_read_readvariableop-savev2_encoder2_fcn1_bias_read_readvariableop/savev2_encoder1_fcn1_kernel_read_readvariableop-savev2_encoder1_fcn1_bias_read_readvariableop4savev2_encoder1_fcn3_gene_kernel_read_readvariableop2savev2_encoder1_fcn3_gene_bias_read_readvariableop4savev2_encoder2_fcn2_mean_kernel_read_readvariableop2savev2_encoder2_fcn2_mean_bias_read_readvariableop4savev2_encoder1_fcn2_mean_kernel_read_readvariableop2savev2_encoder1_fcn2_mean_bias_read_readvariableop5savev2_decoder1_layer_gene_kernel_read_readvariableop3savev2_decoder1_layer_gene_bias_read_readvariableop/savev2_decoder1_fcn1_kernel_read_readvariableop-savev2_decoder1_fcn1_bias_read_readvariableop/savev2_decoder2_fcn1_kernel_read_readvariableop-savev2_decoder2_fcn1_bias_read_readvariableop/savev2_decoder1_fcn2_kernel_read_readvariableop-savev2_decoder1_fcn2_bias_read_readvariableop/savev2_decoder2_fcn2_kernel_read_readvariableop-savev2_decoder2_fcn2_bias_read_readvariableop/savev2_encoder1_fcn3_kernel_read_readvariableop-savev2_encoder1_fcn3_bias_read_readvariableop/savev2_encoder2_fcn3_kernel_read_readvariableop-savev2_encoder2_fcn3_bias_read_readvariableopsavev2_const_1"/device:CPU:0*
_output_shapes
 *'
dtypes
22
SaveV2║
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:2(
&MergeV2Checkpoints/checkpoint_prefixesб
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*
_output_shapes
 2
MergeV2Checkpointsr
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: 2

Identitym

Identity_1IdentityIdentity:output:0^MergeV2Checkpoints*
T0*
_output_shapes
: 2

Identity_1"!

identity_1Identity_1:output:0*ц
_input_shapes╘
╤: :	Аx:x:	Аd:d:	Аd:d:x`:`:dd:d:dd:d:	dА	:А	:	`А	:А	:
А	А:А:
А	А:А:dP:P:`P:P: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:%!

_output_shapes
:	Аx: 

_output_shapes
:x:%!

_output_shapes
:	Аd: 

_output_shapes
:d:%!

_output_shapes
:	Аd: 

_output_shapes
:d:$ 

_output_shapes

:x`: 

_output_shapes
:`:$	 

_output_shapes

:dd: 


_output_shapes
:d:$ 

_output_shapes

:dd: 

_output_shapes
:d:%!

_output_shapes
:	dА	:!

_output_shapes	
:А	:%!

_output_shapes
:	`А	:!

_output_shapes	
:А	:&"
 
_output_shapes
:
А	А:!

_output_shapes	
:А:&"
 
_output_shapes
:
А	А:!

_output_shapes	
:А:$ 

_output_shapes

:dP: 

_output_shapes
:P:$ 

_output_shapes

:`P: 

_output_shapes
:P:

_output_shapes
: 
╔В
Ф
C__inference_model_layer_call_and_return_conditional_losses_40051263
inputs_0
inputs_1
inputs_2
inputs_3
inputs_4
inputs_5?
,encoder2_fcn1_matmul_readvariableop_resource:	Аx;
-encoder2_fcn1_biasadd_readvariableop_resource:xC
1encoder2_fcn2_mean_matmul_readvariableop_resource:x`@
2encoder2_fcn2_mean_biasadd_readvariableop_resource:`D
1encoder1_fcn3_gene_matmul_readvariableop_resource:	Аd@
2encoder1_fcn3_gene_biasadd_readvariableop_resource:d?
,encoder1_fcn1_matmul_readvariableop_resource:	Аd;
-encoder1_fcn1_biasadd_readvariableop_resource:dC
1encoder1_fcn2_mean_matmul_readvariableop_resource:dd@
2encoder1_fcn2_mean_biasadd_readvariableop_resource:dD
2decoder1_layer_gene_matmul_readvariableop_resource:ddA
3decoder1_layer_gene_biasadd_readvariableop_resource:d?
,decoder2_fcn1_matmul_readvariableop_resource:	`А	<
-decoder2_fcn1_biasadd_readvariableop_resource:	А	?
,decoder1_fcn1_matmul_readvariableop_resource:	dА	<
-decoder1_fcn1_biasadd_readvariableop_resource:	А	@
,decoder2_fcn2_matmul_readvariableop_resource:
А	А<
-decoder2_fcn2_biasadd_readvariableop_resource:	А@
,decoder1_fcn2_matmul_readvariableop_resource:
А	А<
-decoder1_fcn2_biasadd_readvariableop_resource:	А>
,encoder2_fcn3_matmul_readvariableop_resource:`P;
-encoder2_fcn3_biasadd_readvariableop_resource:P>
,encoder1_fcn3_matmul_readvariableop_resource:dP;
-encoder1_fcn3_biasadd_readvariableop_resource:P
unknown
identity

identity_1

identity_2

identity_3Ив$decoder1/fcn1/BiasAdd/ReadVariableOpв&decoder1/fcn1/BiasAdd_1/ReadVariableOpв#decoder1/fcn1/MatMul/ReadVariableOpв%decoder1/fcn1/MatMul_1/ReadVariableOpв$decoder1/fcn2/BiasAdd/ReadVariableOpв&decoder1/fcn2/BiasAdd_1/ReadVariableOpв#decoder1/fcn2/MatMul/ReadVariableOpв%decoder1/fcn2/MatMul_1/ReadVariableOpв*decoder1/layer_gene/BiasAdd/ReadVariableOpв,decoder1/layer_gene/BiasAdd_1/ReadVariableOpв)decoder1/layer_gene/MatMul/ReadVariableOpв+decoder1/layer_gene/MatMul_1/ReadVariableOpв$decoder2/fcn1/BiasAdd/ReadVariableOpв&decoder2/fcn1/BiasAdd_1/ReadVariableOpв#decoder2/fcn1/MatMul/ReadVariableOpв%decoder2/fcn1/MatMul_1/ReadVariableOpв$decoder2/fcn2/BiasAdd/ReadVariableOpв&decoder2/fcn2/BiasAdd_1/ReadVariableOpв#decoder2/fcn2/MatMul/ReadVariableOpв%decoder2/fcn2/MatMul_1/ReadVariableOpв$encoder1/fcn1/BiasAdd/ReadVariableOpв&encoder1/fcn1/BiasAdd_1/ReadVariableOpв#encoder1/fcn1/MatMul/ReadVariableOpв%encoder1/fcn1/MatMul_1/ReadVariableOpв)encoder1/fcn2/mean/BiasAdd/ReadVariableOpв+encoder1/fcn2/mean/BiasAdd_1/ReadVariableOpв(encoder1/fcn2/mean/MatMul/ReadVariableOpв*encoder1/fcn2/mean/MatMul_1/ReadVariableOpв$encoder1/fcn3/BiasAdd/ReadVariableOpв#encoder1/fcn3/MatMul/ReadVariableOpв)encoder1/fcn3_gene/BiasAdd/ReadVariableOpв+encoder1/fcn3_gene/BiasAdd_1/ReadVariableOpв(encoder1/fcn3_gene/MatMul/ReadVariableOpв*encoder1/fcn3_gene/MatMul_1/ReadVariableOpв$encoder2/fcn1/BiasAdd/ReadVariableOpв&encoder2/fcn1/BiasAdd_1/ReadVariableOpв#encoder2/fcn1/MatMul/ReadVariableOpв%encoder2/fcn1/MatMul_1/ReadVariableOpв)encoder2/fcn2/mean/BiasAdd/ReadVariableOpв+encoder2/fcn2/mean/BiasAdd_1/ReadVariableOpв(encoder2/fcn2/mean/MatMul/ReadVariableOpв*encoder2/fcn2/mean/MatMul_1/ReadVariableOpв$encoder2/fcn3/BiasAdd/ReadVariableOpв#encoder2/fcn3/MatMul/ReadVariableOp╕
#encoder2/fcn1/MatMul/ReadVariableOpReadVariableOp,encoder2_fcn1_matmul_readvariableop_resource*
_output_shapes
:	Аx*
dtype02%
#encoder2/fcn1/MatMul/ReadVariableOpЯ
encoder2/fcn1/MatMulMatMulinputs_1+encoder2/fcn1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         x2
encoder2/fcn1/MatMul╢
$encoder2/fcn1/BiasAdd/ReadVariableOpReadVariableOp-encoder2_fcn1_biasadd_readvariableop_resource*
_output_shapes
:x*
dtype02&
$encoder2/fcn1/BiasAdd/ReadVariableOp╣
encoder2/fcn1/BiasAddBiasAddencoder2/fcn1/MatMul:product:0,encoder2/fcn1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         x2
encoder2/fcn1/BiasAddВ
encoder2/fcn1/ReluReluencoder2/fcn1/BiasAdd:output:0*
T0*'
_output_shapes
:         x2
encoder2/fcn1/Relu╞
(encoder2/fcn2/mean/MatMul/ReadVariableOpReadVariableOp1encoder2_fcn2_mean_matmul_readvariableop_resource*
_output_shapes

:x`*
dtype02*
(encoder2/fcn2/mean/MatMul/ReadVariableOp╞
encoder2/fcn2/mean/MatMulMatMul encoder2/fcn1/Relu:activations:00encoder2/fcn2/mean/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         `2
encoder2/fcn2/mean/MatMul┼
)encoder2/fcn2/mean/BiasAdd/ReadVariableOpReadVariableOp2encoder2_fcn2_mean_biasadd_readvariableop_resource*
_output_shapes
:`*
dtype02+
)encoder2/fcn2/mean/BiasAdd/ReadVariableOp═
encoder2/fcn2/mean/BiasAddBiasAdd#encoder2/fcn2/mean/MatMul:product:01encoder2/fcn2/mean/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         `2
encoder2/fcn2/mean/BiasAddЧ
encoder2/fcn2/mean/LeakyRelu	LeakyRelu#encoder2/fcn2/mean/BiasAdd:output:0*'
_output_shapes
:         `2
encoder2/fcn2/mean/LeakyRelu╟
(encoder1/fcn3_gene/MatMul/ReadVariableOpReadVariableOp1encoder1_fcn3_gene_matmul_readvariableop_resource*
_output_shapes
:	Аd*
dtype02*
(encoder1/fcn3_gene/MatMul/ReadVariableOpо
encoder1/fcn3_gene/MatMulMatMulinputs_30encoder1/fcn3_gene/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         d2
encoder1/fcn3_gene/MatMul┼
)encoder1/fcn3_gene/BiasAdd/ReadVariableOpReadVariableOp2encoder1_fcn3_gene_biasadd_readvariableop_resource*
_output_shapes
:d*
dtype02+
)encoder1/fcn3_gene/BiasAdd/ReadVariableOp═
encoder1/fcn3_gene/BiasAddBiasAdd#encoder1/fcn3_gene/MatMul:product:01encoder1/fcn3_gene/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         d2
encoder1/fcn3_gene/BiasAdd╕
#encoder1/fcn1/MatMul/ReadVariableOpReadVariableOp,encoder1_fcn1_matmul_readvariableop_resource*
_output_shapes
:	Аd*
dtype02%
#encoder1/fcn1/MatMul/ReadVariableOpЯ
encoder1/fcn1/MatMulMatMulinputs_2+encoder1/fcn1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         d2
encoder1/fcn1/MatMul╢
$encoder1/fcn1/BiasAdd/ReadVariableOpReadVariableOp-encoder1_fcn1_biasadd_readvariableop_resource*
_output_shapes
:d*
dtype02&
$encoder1/fcn1/BiasAdd/ReadVariableOp╣
encoder1/fcn1/BiasAddBiasAddencoder1/fcn1/MatMul:product:0,encoder1/fcn1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         d2
encoder1/fcn1/BiasAddВ
encoder1/fcn1/ReluReluencoder1/fcn1/BiasAdd:output:0*
T0*'
_output_shapes
:         d2
encoder1/fcn1/Relu╦
*encoder1/fcn3_gene/MatMul_1/ReadVariableOpReadVariableOp1encoder1_fcn3_gene_matmul_readvariableop_resource*
_output_shapes
:	Аd*
dtype02,
*encoder1/fcn3_gene/MatMul_1/ReadVariableOp┤
encoder1/fcn3_gene/MatMul_1MatMulinputs_52encoder1/fcn3_gene/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:         d2
encoder1/fcn3_gene/MatMul_1╔
+encoder1/fcn3_gene/BiasAdd_1/ReadVariableOpReadVariableOp2encoder1_fcn3_gene_biasadd_readvariableop_resource*
_output_shapes
:d*
dtype02-
+encoder1/fcn3_gene/BiasAdd_1/ReadVariableOp╒
encoder1/fcn3_gene/BiasAdd_1BiasAdd%encoder1/fcn3_gene/MatMul_1:product:03encoder1/fcn3_gene/BiasAdd_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:         d2
encoder1/fcn3_gene/BiasAdd_1╝
%encoder1/fcn1/MatMul_1/ReadVariableOpReadVariableOp,encoder1_fcn1_matmul_readvariableop_resource*
_output_shapes
:	Аd*
dtype02'
%encoder1/fcn1/MatMul_1/ReadVariableOpе
encoder1/fcn1/MatMul_1MatMulinputs_0-encoder1/fcn1/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:         d2
encoder1/fcn1/MatMul_1║
&encoder1/fcn1/BiasAdd_1/ReadVariableOpReadVariableOp-encoder1_fcn1_biasadd_readvariableop_resource*
_output_shapes
:d*
dtype02(
&encoder1/fcn1/BiasAdd_1/ReadVariableOp┴
encoder1/fcn1/BiasAdd_1BiasAdd encoder1/fcn1/MatMul_1:product:0.encoder1/fcn1/BiasAdd_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:         d2
encoder1/fcn1/BiasAdd_1И
encoder1/fcn1/Relu_1Relu encoder1/fcn1/BiasAdd_1:output:0*
T0*'
_output_shapes
:         d2
encoder1/fcn1/Relu_1Ъ
tf.math.square_5/SquareSquare*encoder2/fcn2/mean/LeakyRelu:activations:0*
T0*'
_output_shapes
:         `2
tf.math.square_5/SquareЪ
tf.math.square_4/SquareSquare*encoder2/fcn2/mean/LeakyRelu:activations:0*
T0*'
_output_shapes
:         `2
tf.math.square_4/Squarey
tf.math.square_3/SquareSquareinputs_1*
T0*(
_output_shapes
:         А2
tf.math.square_3/Squarey
tf.math.square_2/SquareSquareinputs_1*
T0*(
_output_shapes
:         А2
tf.math.square_2/Square░
tf.math.subtract_3/SubSub encoder1/fcn1/Relu:activations:0#encoder1/fcn3_gene/BiasAdd:output:0*
T0*'
_output_shapes
:         d2
tf.math.subtract_3/Sub░
tf.math.subtract/SubSub"encoder1/fcn1/Relu_1:activations:0%encoder1/fcn3_gene/BiasAdd_1:output:0*
T0*'
_output_shapes
:         d2
tf.math.subtract/Subг
*tf.math.reduce_sum_3/Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
valueB :
         2,
*tf.math.reduce_sum_3/Sum/reduction_indices╨
tf.math.reduce_sum_3/SumSumtf.math.square_5/Square:y:03tf.math.reduce_sum_3/Sum/reduction_indices:output:0*
T0*'
_output_shapes
:         *
	keep_dims(2
tf.math.reduce_sum_3/Sumг
*tf.math.reduce_sum_2/Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
valueB :
         2,
*tf.math.reduce_sum_2/Sum/reduction_indices╨
tf.math.reduce_sum_2/SumSumtf.math.square_4/Square:y:03tf.math.reduce_sum_2/Sum/reduction_indices:output:0*
T0*'
_output_shapes
:         *
	keep_dims(2
tf.math.reduce_sum_2/Sumг
*tf.math.reduce_sum_1/Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
valueB :
         2,
*tf.math.reduce_sum_1/Sum/reduction_indices╨
tf.math.reduce_sum_1/SumSumtf.math.square_3/Square:y:03tf.math.reduce_sum_1/Sum/reduction_indices:output:0*
T0*'
_output_shapes
:         *
	keep_dims(2
tf.math.reduce_sum_1/SumЯ
(tf.math.reduce_sum/Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
valueB :
         2*
(tf.math.reduce_sum/Sum/reduction_indices╩
tf.math.reduce_sum/SumSumtf.math.square_2/Square:y:01tf.math.reduce_sum/Sum/reduction_indices:output:0*
T0*'
_output_shapes
:         *
	keep_dims(2
tf.math.reduce_sum/Sum╞
(encoder1/fcn2/mean/MatMul/ReadVariableOpReadVariableOp1encoder1_fcn2_mean_matmul_readvariableop_resource*
_output_shapes

:dd*
dtype02*
(encoder1/fcn2/mean/MatMul/ReadVariableOp└
encoder1/fcn2/mean/MatMulMatMultf.math.subtract_3/Sub:z:00encoder1/fcn2/mean/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         d2
encoder1/fcn2/mean/MatMul┼
)encoder1/fcn2/mean/BiasAdd/ReadVariableOpReadVariableOp2encoder1_fcn2_mean_biasadd_readvariableop_resource*
_output_shapes
:d*
dtype02+
)encoder1/fcn2/mean/BiasAdd/ReadVariableOp═
encoder1/fcn2/mean/BiasAddBiasAdd#encoder1/fcn2/mean/MatMul:product:01encoder1/fcn2/mean/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         d2
encoder1/fcn2/mean/BiasAddЧ
encoder1/fcn2/mean/LeakyRelu	LeakyRelu#encoder1/fcn2/mean/BiasAdd:output:0*'
_output_shapes
:         d2
encoder1/fcn2/mean/LeakyRelu╩
*encoder1/fcn2/mean/MatMul_1/ReadVariableOpReadVariableOp1encoder1_fcn2_mean_matmul_readvariableop_resource*
_output_shapes

:dd*
dtype02,
*encoder1/fcn2/mean/MatMul_1/ReadVariableOp─
encoder1/fcn2/mean/MatMul_1MatMultf.math.subtract/Sub:z:02encoder1/fcn2/mean/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:         d2
encoder1/fcn2/mean/MatMul_1╔
+encoder1/fcn2/mean/BiasAdd_1/ReadVariableOpReadVariableOp2encoder1_fcn2_mean_biasadd_readvariableop_resource*
_output_shapes
:d*
dtype02-
+encoder1/fcn2/mean/BiasAdd_1/ReadVariableOp╒
encoder1/fcn2/mean/BiasAdd_1BiasAdd%encoder1/fcn2/mean/MatMul_1:product:03encoder1/fcn2/mean/BiasAdd_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:         d2
encoder1/fcn2/mean/BiasAdd_1Э
encoder1/fcn2/mean/LeakyRelu_1	LeakyRelu%encoder1/fcn2/mean/BiasAdd_1:output:0*'
_output_shapes
:         d2 
encoder1/fcn2/mean/LeakyRelu_1З
tf.math.sqrt_2/SqrtSqrt!tf.math.reduce_sum_2/Sum:output:0*
T0*'
_output_shapes
:         2
tf.math.sqrt_2/SqrtЗ
tf.math.sqrt_3/SqrtSqrt!tf.math.reduce_sum_3/Sum:output:0*
T0*'
_output_shapes
:         2
tf.math.sqrt_3/SqrtБ
tf.math.sqrt/SqrtSqrttf.math.reduce_sum/Sum:output:0*
T0*'
_output_shapes
:         2
tf.math.sqrt/SqrtЗ
tf.math.sqrt_1/SqrtSqrt!tf.math.reduce_sum_1/Sum:output:0*
T0*'
_output_shapes
:         2
tf.math.sqrt_1/Sqrt╝
%encoder2/fcn1/MatMul_1/ReadVariableOpReadVariableOp,encoder2_fcn1_matmul_readvariableop_resource*
_output_shapes
:	Аx*
dtype02'
%encoder2/fcn1/MatMul_1/ReadVariableOpе
encoder2/fcn1/MatMul_1MatMulinputs_4-encoder2/fcn1/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:         x2
encoder2/fcn1/MatMul_1║
&encoder2/fcn1/BiasAdd_1/ReadVariableOpReadVariableOp-encoder2_fcn1_biasadd_readvariableop_resource*
_output_shapes
:x*
dtype02(
&encoder2/fcn1/BiasAdd_1/ReadVariableOp┴
encoder2/fcn1/BiasAdd_1BiasAdd encoder2/fcn1/MatMul_1:product:0.encoder2/fcn1/BiasAdd_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:         x2
encoder2/fcn1/BiasAdd_1И
encoder2/fcn1/Relu_1Relu encoder2/fcn1/BiasAdd_1:output:0*
T0*'
_output_shapes
:         x2
encoder2/fcn1/Relu_1╔
)decoder1/layer_gene/MatMul/ReadVariableOpReadVariableOp2decoder1_layer_gene_matmul_readvariableop_resource*
_output_shapes

:dd*
dtype02+
)decoder1/layer_gene/MatMul/ReadVariableOp╙
decoder1/layer_gene/MatMulMatMul*encoder1/fcn2/mean/LeakyRelu:activations:01decoder1/layer_gene/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         d2
decoder1/layer_gene/MatMul╚
*decoder1/layer_gene/BiasAdd/ReadVariableOpReadVariableOp3decoder1_layer_gene_biasadd_readvariableop_resource*
_output_shapes
:d*
dtype02,
*decoder1/layer_gene/BiasAdd/ReadVariableOp╤
decoder1/layer_gene/BiasAddBiasAdd$decoder1/layer_gene/MatMul:product:02decoder1/layer_gene/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         d2
decoder1/layer_gene/BiasAddЪ
decoder1/layer_gene/LeakyRelu	LeakyRelu$decoder1/layer_gene/BiasAdd:output:0*'
_output_shapes
:         d2
decoder1/layer_gene/LeakyRelu═
+decoder1/layer_gene/MatMul_1/ReadVariableOpReadVariableOp2decoder1_layer_gene_matmul_readvariableop_resource*
_output_shapes

:dd*
dtype02-
+decoder1/layer_gene/MatMul_1/ReadVariableOp█
decoder1/layer_gene/MatMul_1MatMul,encoder1/fcn2/mean/LeakyRelu_1:activations:03decoder1/layer_gene/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:         d2
decoder1/layer_gene/MatMul_1╠
,decoder1/layer_gene/BiasAdd_1/ReadVariableOpReadVariableOp3decoder1_layer_gene_biasadd_readvariableop_resource*
_output_shapes
:d*
dtype02.
,decoder1/layer_gene/BiasAdd_1/ReadVariableOp┘
decoder1/layer_gene/BiasAdd_1BiasAdd&decoder1/layer_gene/MatMul_1:product:04decoder1/layer_gene/BiasAdd_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:         d2
decoder1/layer_gene/BiasAdd_1а
decoder1/layer_gene/LeakyRelu_1	LeakyRelu&decoder1/layer_gene/BiasAdd_1:output:0*'
_output_shapes
:         d2!
decoder1/layer_gene/LeakyRelu_1ц
tf.linalg.matmul_1/MatMulMatMul*encoder2/fcn2/mean/LeakyRelu:activations:0*encoder2/fcn2/mean/LeakyRelu:activations:0*
T0*0
_output_shapes
:                  *
transpose_b(2
tf.linalg.matmul_1/MatMulЫ
tf.math.multiply_3/MulMultf.math.sqrt_2/Sqrt:y:0tf.math.sqrt_3/Sqrt:y:0*
T0*'
_output_shapes
:         2
tf.math.multiply_3/MulЮ
tf.linalg.matmul/MatMulMatMulinputs_1inputs_1*
T0*0
_output_shapes
:                  *
transpose_b(2
tf.linalg.matmul/MatMulЩ
tf.math.multiply_2/MulMultf.math.sqrt/Sqrt:y:0tf.math.sqrt_1/Sqrt:y:0*
T0*'
_output_shapes
:         2
tf.math.multiply_2/Mul╩
*encoder2/fcn2/mean/MatMul_1/ReadVariableOpReadVariableOp1encoder2_fcn2_mean_matmul_readvariableop_resource*
_output_shapes

:x`*
dtype02,
*encoder2/fcn2/mean/MatMul_1/ReadVariableOp╬
encoder2/fcn2/mean/MatMul_1MatMul"encoder2/fcn1/Relu_1:activations:02encoder2/fcn2/mean/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:         `2
encoder2/fcn2/mean/MatMul_1╔
+encoder2/fcn2/mean/BiasAdd_1/ReadVariableOpReadVariableOp2encoder2_fcn2_mean_biasadd_readvariableop_resource*
_output_shapes
:`*
dtype02-
+encoder2/fcn2/mean/BiasAdd_1/ReadVariableOp╒
encoder2/fcn2/mean/BiasAdd_1BiasAdd%encoder2/fcn2/mean/MatMul_1:product:03encoder2/fcn2/mean/BiasAdd_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:         `2
encoder2/fcn2/mean/BiasAdd_1Э
encoder2/fcn2/mean/LeakyRelu_1	LeakyRelu%encoder2/fcn2/mean/BiasAdd_1:output:0*'
_output_shapes
:         `2 
encoder2/fcn2/mean/LeakyRelu_1╔
tf.__operators__.add_2/AddV2AddV2+decoder1/layer_gene/LeakyRelu:activations:0#encoder1/fcn3_gene/BiasAdd:output:0*
T0*'
_output_shapes
:         d2
tf.__operators__.add_2/AddV2╔
tf.__operators__.add/AddV2AddV2-decoder1/layer_gene/LeakyRelu_1:activations:0%encoder1/fcn3_gene/BiasAdd_1:output:0*
T0*'
_output_shapes
:         d2
tf.__operators__.add/AddV2╜
tf.math.truediv_1/truedivRealDiv#tf.linalg.matmul_1/MatMul:product:0tf.math.multiply_3/Mul:z:0*
T0*0
_output_shapes
:                  2
tf.math.truediv_1/truediv╖
tf.math.truediv/truedivRealDiv!tf.linalg.matmul/MatMul:product:0tf.math.multiply_2/Mul:z:0*
T0*0
_output_shapes
:                  2
tf.math.truediv/truediv╕
#decoder2/fcn1/MatMul/ReadVariableOpReadVariableOp,decoder2_fcn1_matmul_readvariableop_resource*
_output_shapes
:	`А	*
dtype02%
#decoder2/fcn1/MatMul/ReadVariableOp─
decoder2/fcn1/MatMulMatMul,encoder2/fcn2/mean/LeakyRelu_1:activations:0+decoder2/fcn1/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         А	2
decoder2/fcn1/MatMul╖
$decoder2/fcn1/BiasAdd/ReadVariableOpReadVariableOp-decoder2_fcn1_biasadd_readvariableop_resource*
_output_shapes	
:А	*
dtype02&
$decoder2/fcn1/BiasAdd/ReadVariableOp║
decoder2/fcn1/BiasAddBiasAdddecoder2/fcn1/MatMul:product:0,decoder2/fcn1/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         А	2
decoder2/fcn1/BiasAddЙ
decoder2/fcn1/LeakyRelu	LeakyReludecoder2/fcn1/BiasAdd:output:0*(
_output_shapes
:         А	2
decoder2/fcn1/LeakyRelu╕
#decoder1/fcn1/MatMul/ReadVariableOpReadVariableOp,decoder1_fcn1_matmul_readvariableop_resource*
_output_shapes
:	dА	*
dtype02%
#decoder1/fcn1/MatMul/ReadVariableOp╕
decoder1/fcn1/MatMulMatMul tf.__operators__.add_2/AddV2:z:0+decoder1/fcn1/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         А	2
decoder1/fcn1/MatMul╖
$decoder1/fcn1/BiasAdd/ReadVariableOpReadVariableOp-decoder1_fcn1_biasadd_readvariableop_resource*
_output_shapes	
:А	*
dtype02&
$decoder1/fcn1/BiasAdd/ReadVariableOp║
decoder1/fcn1/BiasAddBiasAdddecoder1/fcn1/MatMul:product:0,decoder1/fcn1/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         А	2
decoder1/fcn1/BiasAddЙ
decoder1/fcn1/LeakyRelu	LeakyReludecoder1/fcn1/BiasAdd:output:0*(
_output_shapes
:         А	2
decoder1/fcn1/LeakyRelu╝
%decoder2/fcn1/MatMul_1/ReadVariableOpReadVariableOp,decoder2_fcn1_matmul_readvariableop_resource*
_output_shapes
:	`А	*
dtype02'
%decoder2/fcn1/MatMul_1/ReadVariableOp╚
decoder2/fcn1/MatMul_1MatMul*encoder2/fcn2/mean/LeakyRelu:activations:0-decoder2/fcn1/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:         А	2
decoder2/fcn1/MatMul_1╗
&decoder2/fcn1/BiasAdd_1/ReadVariableOpReadVariableOp-decoder2_fcn1_biasadd_readvariableop_resource*
_output_shapes	
:А	*
dtype02(
&decoder2/fcn1/BiasAdd_1/ReadVariableOp┬
decoder2/fcn1/BiasAdd_1BiasAdd decoder2/fcn1/MatMul_1:product:0.decoder2/fcn1/BiasAdd_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:         А	2
decoder2/fcn1/BiasAdd_1П
decoder2/fcn1/LeakyRelu_1	LeakyRelu decoder2/fcn1/BiasAdd_1:output:0*(
_output_shapes
:         А	2
decoder2/fcn1/LeakyRelu_1╝
%decoder1/fcn1/MatMul_1/ReadVariableOpReadVariableOp,decoder1_fcn1_matmul_readvariableop_resource*
_output_shapes
:	dА	*
dtype02'
%decoder1/fcn1/MatMul_1/ReadVariableOp╝
decoder1/fcn1/MatMul_1MatMultf.__operators__.add/AddV2:z:0-decoder1/fcn1/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:         А	2
decoder1/fcn1/MatMul_1╗
&decoder1/fcn1/BiasAdd_1/ReadVariableOpReadVariableOp-decoder1_fcn1_biasadd_readvariableop_resource*
_output_shapes	
:А	*
dtype02(
&decoder1/fcn1/BiasAdd_1/ReadVariableOp┬
decoder1/fcn1/BiasAdd_1BiasAdd decoder1/fcn1/MatMul_1:product:0.decoder1/fcn1/BiasAdd_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:         А	2
decoder1/fcn1/BiasAdd_1П
decoder1/fcn1/LeakyRelu_1	LeakyRelu decoder1/fcn1/BiasAdd_1:output:0*(
_output_shapes
:         А	2
decoder1/fcn1/LeakyRelu_1Ч
tf.nn.softmax_1/SoftmaxSoftmaxtf.math.truediv_1/truediv:z:0*
T0*0
_output_shapes
:                  2
tf.nn.softmax_1/SoftmaxС
tf.nn.softmax/SoftmaxSoftmaxtf.math.truediv/truediv:z:0*
T0*0
_output_shapes
:                  2
tf.nn.softmax/Softmaxy
tf.math.multiply_9/Mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *  HB2
tf.math.multiply_9/Mul/yЧ
tf.math.multiply_9/MulMulinputs_4!tf.math.multiply_9/Mul/y:output:0*
T0*(
_output_shapes
:         А2
tf.math.multiply_9/Mul╣
#decoder2/fcn2/MatMul/ReadVariableOpReadVariableOp,decoder2_fcn2_matmul_readvariableop_resource* 
_output_shapes
:
А	А*
dtype02%
#decoder2/fcn2/MatMul/ReadVariableOp╜
decoder2/fcn2/MatMulMatMul%decoder2/fcn1/LeakyRelu:activations:0+decoder2/fcn2/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         А2
decoder2/fcn2/MatMul╖
$decoder2/fcn2/BiasAdd/ReadVariableOpReadVariableOp-decoder2_fcn2_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype02&
$decoder2/fcn2/BiasAdd/ReadVariableOp║
decoder2/fcn2/BiasAddBiasAdddecoder2/fcn2/MatMul:product:0,decoder2/fcn2/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         А2
decoder2/fcn2/BiasAddy
tf.math.multiply_8/Mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *  HB2
tf.math.multiply_8/Mul/yЧ
tf.math.multiply_8/MulMulinputs_2!tf.math.multiply_8/Mul/y:output:0*
T0*(
_output_shapes
:         А2
tf.math.multiply_8/Mul╣
#decoder1/fcn2/MatMul/ReadVariableOpReadVariableOp,decoder1_fcn2_matmul_readvariableop_resource* 
_output_shapes
:
А	А*
dtype02%
#decoder1/fcn2/MatMul/ReadVariableOp╜
decoder1/fcn2/MatMulMatMul%decoder1/fcn1/LeakyRelu:activations:0+decoder1/fcn2/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         А2
decoder1/fcn2/MatMul╖
$decoder1/fcn2/BiasAdd/ReadVariableOpReadVariableOp-decoder1_fcn2_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype02&
$decoder1/fcn2/BiasAdd/ReadVariableOp║
decoder1/fcn2/BiasAddBiasAdddecoder1/fcn2/MatMul:product:0,decoder1/fcn2/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         А2
decoder1/fcn2/BiasAdd╜
%decoder2/fcn2/MatMul_1/ReadVariableOpReadVariableOp,decoder2_fcn2_matmul_readvariableop_resource* 
_output_shapes
:
А	А*
dtype02'
%decoder2/fcn2/MatMul_1/ReadVariableOp┼
decoder2/fcn2/MatMul_1MatMul'decoder2/fcn1/LeakyRelu_1:activations:0-decoder2/fcn2/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:         А2
decoder2/fcn2/MatMul_1╗
&decoder2/fcn2/BiasAdd_1/ReadVariableOpReadVariableOp-decoder2_fcn2_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype02(
&decoder2/fcn2/BiasAdd_1/ReadVariableOp┬
decoder2/fcn2/BiasAdd_1BiasAdd decoder2/fcn2/MatMul_1:product:0.decoder2/fcn2/BiasAdd_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:         А2
decoder2/fcn2/BiasAdd_1╜
%decoder1/fcn2/MatMul_1/ReadVariableOpReadVariableOp,decoder1_fcn2_matmul_readvariableop_resource* 
_output_shapes
:
А	А*
dtype02'
%decoder1/fcn2/MatMul_1/ReadVariableOp┼
decoder1/fcn2/MatMul_1MatMul'decoder1/fcn1/LeakyRelu_1:activations:0-decoder1/fcn2/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:         А2
decoder1/fcn2/MatMul_1╗
&decoder1/fcn2/BiasAdd_1/ReadVariableOpReadVariableOp-decoder1_fcn2_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype02(
&decoder1/fcn2/BiasAdd_1/ReadVariableOp┬
decoder1/fcn2/BiasAdd_1BiasAdd decoder1/fcn2/MatMul_1:product:0.decoder1/fcn2/BiasAdd_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:         А2
decoder1/fcn2/BiasAdd_1└
tf.math.truediv_3/truedivRealDiv!tf.nn.softmax_1/Softmax:softmax:0tf.nn.softmax/Softmax:softmax:0*
T0*0
_output_shapes
:                  2
tf.math.truediv_3/truediv└
tf.math.truediv_2/truedivRealDivtf.nn.softmax/Softmax:softmax:0!tf.nn.softmax_1/Softmax:softmax:0*
T0*0
_output_shapes
:                  2
tf.math.truediv_2/truedivж
tf.math.subtract_5/SubSubtf.math.multiply_9/Mul:z:0decoder2/fcn2/BiasAdd:output:0*
T0*(
_output_shapes
:         А2
tf.math.subtract_5/Subж
tf.math.subtract_4/SubSubtf.math.multiply_8/Mul:z:0decoder1/fcn2/BiasAdd:output:0*
T0*(
_output_shapes
:         А2
tf.math.subtract_4/Suby
tf.math.multiply_1/Mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *  HB2
tf.math.multiply_1/Mul/yЧ
tf.math.multiply_1/MulMulinputs_1!tf.math.multiply_1/Mul/y:output:0*
T0*(
_output_shapes
:         А2
tf.math.multiply_1/Mulu
tf.math.multiply/Mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *  HB2
tf.math.multiply/Mul/yС
tf.math.multiply/MulMulinputs_0tf.math.multiply/Mul/y:output:0*
T0*(
_output_shapes
:         А2
tf.math.multiply/MulЗ
tf.math.log_1/LogLogtf.math.truediv_3/truediv:z:0*
T0*0
_output_shapes
:                  2
tf.math.log_1/LogГ
tf.math.log/LogLogtf.math.truediv_2/truediv:z:0*
T0*0
_output_shapes
:                  2
tf.math.log/LogЛ
tf.math.square_9/SquareSquaretf.math.subtract_5/Sub:z:0*
T0*(
_output_shapes
:         А2
tf.math.square_9/SquareЛ
tf.math.square_8/SquareSquaretf.math.subtract_4/Sub:z:0*
T0*(
_output_shapes
:         А2
tf.math.square_8/Squareи
tf.math.subtract_2/SubSubtf.math.multiply_1/Mul:z:0 decoder2/fcn2/BiasAdd_1:output:0*
T0*(
_output_shapes
:         А2
tf.math.subtract_2/Subж
tf.math.subtract_1/SubSubtf.math.multiply/Mul:z:0 decoder1/fcn2/BiasAdd_1:output:0*
T0*(
_output_shapes
:         А2
tf.math.subtract_1/Subм
tf.math.multiply_5/MulMul!tf.nn.softmax_1/Softmax:softmax:0tf.math.log_1/Log:y:0*
T0*0
_output_shapes
:                  2
tf.math.multiply_5/Mulи
tf.math.multiply_4/MulMultf.nn.softmax/Softmax:softmax:0tf.math.log/Log:y:0*
T0*0
_output_shapes
:                  2
tf.math.multiply_4/MulЛ
tf.math.reduce_mean_3/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2
tf.math.reduce_mean_3/Constд
tf.math.reduce_mean_3/MeanMeantf.math.square_8/Square:y:0$tf.math.reduce_mean_3/Const:output:0*
T0*
_output_shapes
: 2
tf.math.reduce_mean_3/MeanЛ
tf.math.reduce_mean_4/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2
tf.math.reduce_mean_4/Constд
tf.math.reduce_mean_4/MeanMeantf.math.square_9/Square:y:0$tf.math.reduce_mean_4/Const:output:0*
T0*
_output_shapes
: 2
tf.math.reduce_mean_4/MeanЛ
tf.math.square_1/SquareSquaretf.math.subtract_2/Sub:z:0*
T0*(
_output_shapes
:         А2
tf.math.square_1/SquareЗ
tf.math.square/SquareSquaretf.math.subtract_1/Sub:z:0*
T0*(
_output_shapes
:         А2
tf.math.square/Square╖
#encoder2/fcn3/MatMul/ReadVariableOpReadVariableOp,encoder2_fcn3_matmul_readvariableop_resource*
_output_shapes

:`P*
dtype02%
#encoder2/fcn3/MatMul/ReadVariableOp┴
encoder2/fcn3/MatMulMatMul*encoder2/fcn2/mean/LeakyRelu:activations:0+encoder2/fcn3/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         P2
encoder2/fcn3/MatMul╢
$encoder2/fcn3/BiasAdd/ReadVariableOpReadVariableOp-encoder2_fcn3_biasadd_readvariableop_resource*
_output_shapes
:P*
dtype02&
$encoder2/fcn3/BiasAdd/ReadVariableOp╣
encoder2/fcn3/BiasAddBiasAddencoder2/fcn3/MatMul:product:0,encoder2/fcn3/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         P2
encoder2/fcn3/BiasAdd╖
#encoder1/fcn3/MatMul/ReadVariableOpReadVariableOp,encoder1_fcn3_matmul_readvariableop_resource*
_output_shapes

:dP*
dtype02%
#encoder1/fcn3/MatMul/ReadVariableOp├
encoder1/fcn3/MatMulMatMul,encoder1/fcn2/mean/LeakyRelu_1:activations:0+encoder1/fcn3/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         P2
encoder1/fcn3/MatMul╢
$encoder1/fcn3/BiasAdd/ReadVariableOpReadVariableOp-encoder1_fcn3_biasadd_readvariableop_resource*
_output_shapes
:P*
dtype02&
$encoder1/fcn3/BiasAdd/ReadVariableOp╣
encoder1/fcn3/BiasAddBiasAddencoder1/fcn3/MatMul:product:0,encoder1/fcn3/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         P2
encoder1/fcn3/BiasAddг
*tf.math.reduce_sum_4/Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
valueB :
         2,
*tf.math.reduce_sum_4/Sum/reduction_indices║
tf.math.reduce_sum_4/SumSumtf.math.multiply_4/Mul:z:03tf.math.reduce_sum_4/Sum/reduction_indices:output:0*
T0*#
_output_shapes
:         2
tf.math.reduce_sum_4/Sumг
*tf.math.reduce_sum_5/Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
valueB :
         2,
*tf.math.reduce_sum_5/Sum/reduction_indices║
tf.math.reduce_sum_5/SumSumtf.math.multiply_5/Mul:z:03tf.math.reduce_sum_5/Sum/reduction_indices:output:0*
T0*#
_output_shapes
:         2
tf.math.reduce_sum_5/Sum▓
tf.__operators__.add_16/AddV2AddV2#tf.math.reduce_mean_3/Mean:output:0#tf.math.reduce_mean_4/Mean:output:0*
T0*
_output_shapes
: 2
tf.__operators__.add_16/AddV2З
tf.math.reduce_mean/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2
tf.math.reduce_mean/ConstЬ
tf.math.reduce_mean/MeanMeantf.math.square/Square:y:0"tf.math.reduce_mean/Const:output:0*
T0*
_output_shapes
: 2
tf.math.reduce_mean/MeanЛ
tf.math.reduce_mean_1/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2
tf.math.reduce_mean_1/Constд
tf.math.reduce_mean_1/MeanMeantf.math.square_1/Square:y:0$tf.math.reduce_mean_1/Const:output:0*
T0*
_output_shapes
: 2
tf.math.reduce_mean_1/MeanО
tf.math.square_7/SquareSquareencoder2/fcn3/BiasAdd:output:0*
T0*'
_output_shapes
:         P2
tf.math.square_7/SquareО
tf.math.square_6/SquareSquareencoder1/fcn3/BiasAdd:output:0*
T0*'
_output_shapes
:         P2
tf.math.square_6/Square╣
tf.__operators__.add_1/AddV2AddV2!tf.math.reduce_sum_4/Sum:output:0!tf.math.reduce_sum_5/Sum:output:0*
T0*#
_output_shapes
:         2
tf.__operators__.add_1/AddV2Ж
tf.math.multiply_22/MulMulunknown!tf.__operators__.add_16/AddV2:z:0*
T0*
_output_shapes
: 2
tf.math.multiply_22/Mul░
tf.__operators__.add_15/AddV2AddV2!tf.math.reduce_mean/Mean:output:0#tf.math.reduce_mean_1/Mean:output:0*
T0*
_output_shapes
: 2
tf.__operators__.add_15/AddV2г
*tf.math.reduce_sum_8/Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
valueB :
         2,
*tf.math.reduce_sum_8/Sum/reduction_indices╗
tf.math.reduce_sum_8/SumSumtf.math.square_7/Square:y:03tf.math.reduce_sum_8/Sum/reduction_indices:output:0*
T0*#
_output_shapes
:         2
tf.math.reduce_sum_8/Sumг
*tf.math.reduce_sum_7/Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
valueB :
         2,
*tf.math.reduce_sum_7/Sum/reduction_indices╗
tf.math.reduce_sum_7/SumSumtf.math.square_6/Square:y:03tf.math.reduce_sum_7/Sum/reduction_indices:output:0*
T0*#
_output_shapes
:         2
tf.math.reduce_sum_7/SumД
tf.math.reduce_mean_2/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2
tf.math.reduce_mean_2/Constй
tf.math.reduce_mean_2/MeanMean tf.__operators__.add_1/AddV2:z:0$tf.math.reduce_mean_2/Const:output:0*
T0*
_output_shapes
: 2
tf.math.reduce_mean_2/MeanБ
tf.math.truediv_21/truediv/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
tf.math.truediv_21/truediv/yо
tf.math.truediv_21/truedivRealDiv!tf.__operators__.add_15/AddV2:z:0%tf.math.truediv_21/truediv/y:output:0*
T0*
_output_shapes
: 2
tf.math.truediv_21/truedivБ
tf.math.truediv_22/truediv/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
tf.math.truediv_22/truediv/yи
tf.math.truediv_22/truedivRealDivtf.math.multiply_22/Mul:z:0%tf.math.truediv_22/truediv/y:output:0*
T0*
_output_shapes
: 2
tf.math.truediv_22/truedivГ
tf.math.sqrt_4/SqrtSqrt!tf.math.reduce_sum_7/Sum:output:0*
T0*#
_output_shapes
:         2
tf.math.sqrt_4/SqrtГ
tf.math.sqrt_5/SqrtSqrt!tf.math.reduce_sum_8/Sum:output:0*
T0*#
_output_shapes
:         2
tf.math.sqrt_5/Sqrtй
tf.math.multiply_6/MulMulencoder1/fcn3/BiasAdd:output:0encoder2/fcn3/BiasAdd:output:0*
T0*'
_output_shapes
:         P2
tf.math.multiply_6/Mulи
tf.__operators__.add_17/AddV2AddV2tf.math.truediv_21/truediv:z:0tf.math.truediv_22/truediv:z:0*
T0*
_output_shapes
: 2
tf.__operators__.add_17/AddV2{
tf.math.multiply_23/Mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2
tf.math.multiply_23/Mul/yг
tf.math.multiply_23/MulMul#tf.math.reduce_mean_2/Mean:output:0"tf.math.multiply_23/Mul/y:output:0*
T0*
_output_shapes
: 2
tf.math.multiply_23/Mulг
*tf.math.reduce_sum_6/Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
valueB :
         2,
*tf.math.reduce_sum_6/Sum/reduction_indices║
tf.math.reduce_sum_6/SumSumtf.math.multiply_6/Mul:z:03tf.math.reduce_sum_6/Sum/reduction_indices:output:0*
T0*#
_output_shapes
:         2
tf.math.reduce_sum_6/SumЧ
tf.math.multiply_7/MulMultf.math.sqrt_4/Sqrt:y:0tf.math.sqrt_5/Sqrt:y:0*
T0*#
_output_shapes
:         2
tf.math.multiply_7/Mulи
tf.__operators__.add_18/AddV2AddV2!tf.__operators__.add_17/AddV2:z:0tf.math.multiply_23/Mul:z:0*
T0*
_output_shapes
: 2
tf.__operators__.add_18/AddV2о
tf.math.truediv_4/truedivRealDiv!tf.math.reduce_sum_6/Sum:output:0tf.math.multiply_7/Mul:z:0*
T0*#
_output_shapes
:         2
tf.math.truediv_4/truedivЗ
IdentityIdentitytf.math.truediv_4/truediv:z:0%^decoder1/fcn1/BiasAdd/ReadVariableOp'^decoder1/fcn1/BiasAdd_1/ReadVariableOp$^decoder1/fcn1/MatMul/ReadVariableOp&^decoder1/fcn1/MatMul_1/ReadVariableOp%^decoder1/fcn2/BiasAdd/ReadVariableOp'^decoder1/fcn2/BiasAdd_1/ReadVariableOp$^decoder1/fcn2/MatMul/ReadVariableOp&^decoder1/fcn2/MatMul_1/ReadVariableOp+^decoder1/layer_gene/BiasAdd/ReadVariableOp-^decoder1/layer_gene/BiasAdd_1/ReadVariableOp*^decoder1/layer_gene/MatMul/ReadVariableOp,^decoder1/layer_gene/MatMul_1/ReadVariableOp%^decoder2/fcn1/BiasAdd/ReadVariableOp'^decoder2/fcn1/BiasAdd_1/ReadVariableOp$^decoder2/fcn1/MatMul/ReadVariableOp&^decoder2/fcn1/MatMul_1/ReadVariableOp%^decoder2/fcn2/BiasAdd/ReadVariableOp'^decoder2/fcn2/BiasAdd_1/ReadVariableOp$^decoder2/fcn2/MatMul/ReadVariableOp&^decoder2/fcn2/MatMul_1/ReadVariableOp%^encoder1/fcn1/BiasAdd/ReadVariableOp'^encoder1/fcn1/BiasAdd_1/ReadVariableOp$^encoder1/fcn1/MatMul/ReadVariableOp&^encoder1/fcn1/MatMul_1/ReadVariableOp*^encoder1/fcn2/mean/BiasAdd/ReadVariableOp,^encoder1/fcn2/mean/BiasAdd_1/ReadVariableOp)^encoder1/fcn2/mean/MatMul/ReadVariableOp+^encoder1/fcn2/mean/MatMul_1/ReadVariableOp%^encoder1/fcn3/BiasAdd/ReadVariableOp$^encoder1/fcn3/MatMul/ReadVariableOp*^encoder1/fcn3_gene/BiasAdd/ReadVariableOp,^encoder1/fcn3_gene/BiasAdd_1/ReadVariableOp)^encoder1/fcn3_gene/MatMul/ReadVariableOp+^encoder1/fcn3_gene/MatMul_1/ReadVariableOp%^encoder2/fcn1/BiasAdd/ReadVariableOp'^encoder2/fcn1/BiasAdd_1/ReadVariableOp$^encoder2/fcn1/MatMul/ReadVariableOp&^encoder2/fcn1/MatMul_1/ReadVariableOp*^encoder2/fcn2/mean/BiasAdd/ReadVariableOp,^encoder2/fcn2/mean/BiasAdd_1/ReadVariableOp)^encoder2/fcn2/mean/MatMul/ReadVariableOp+^encoder2/fcn2/mean/MatMul_1/ReadVariableOp%^encoder2/fcn3/BiasAdd/ReadVariableOp$^encoder2/fcn3/MatMul/ReadVariableOp*
T0*#
_output_shapes
:         2

IdentityВ

Identity_1Identity!tf.__operators__.add_18/AddV2:z:0%^decoder1/fcn1/BiasAdd/ReadVariableOp'^decoder1/fcn1/BiasAdd_1/ReadVariableOp$^decoder1/fcn1/MatMul/ReadVariableOp&^decoder1/fcn1/MatMul_1/ReadVariableOp%^decoder1/fcn2/BiasAdd/ReadVariableOp'^decoder1/fcn2/BiasAdd_1/ReadVariableOp$^decoder1/fcn2/MatMul/ReadVariableOp&^decoder1/fcn2/MatMul_1/ReadVariableOp+^decoder1/layer_gene/BiasAdd/ReadVariableOp-^decoder1/layer_gene/BiasAdd_1/ReadVariableOp*^decoder1/layer_gene/MatMul/ReadVariableOp,^decoder1/layer_gene/MatMul_1/ReadVariableOp%^decoder2/fcn1/BiasAdd/ReadVariableOp'^decoder2/fcn1/BiasAdd_1/ReadVariableOp$^decoder2/fcn1/MatMul/ReadVariableOp&^decoder2/fcn1/MatMul_1/ReadVariableOp%^decoder2/fcn2/BiasAdd/ReadVariableOp'^decoder2/fcn2/BiasAdd_1/ReadVariableOp$^decoder2/fcn2/MatMul/ReadVariableOp&^decoder2/fcn2/MatMul_1/ReadVariableOp%^encoder1/fcn1/BiasAdd/ReadVariableOp'^encoder1/fcn1/BiasAdd_1/ReadVariableOp$^encoder1/fcn1/MatMul/ReadVariableOp&^encoder1/fcn1/MatMul_1/ReadVariableOp*^encoder1/fcn2/mean/BiasAdd/ReadVariableOp,^encoder1/fcn2/mean/BiasAdd_1/ReadVariableOp)^encoder1/fcn2/mean/MatMul/ReadVariableOp+^encoder1/fcn2/mean/MatMul_1/ReadVariableOp%^encoder1/fcn3/BiasAdd/ReadVariableOp$^encoder1/fcn3/MatMul/ReadVariableOp*^encoder1/fcn3_gene/BiasAdd/ReadVariableOp,^encoder1/fcn3_gene/BiasAdd_1/ReadVariableOp)^encoder1/fcn3_gene/MatMul/ReadVariableOp+^encoder1/fcn3_gene/MatMul_1/ReadVariableOp%^encoder2/fcn1/BiasAdd/ReadVariableOp'^encoder2/fcn1/BiasAdd_1/ReadVariableOp$^encoder2/fcn1/MatMul/ReadVariableOp&^encoder2/fcn1/MatMul_1/ReadVariableOp*^encoder2/fcn2/mean/BiasAdd/ReadVariableOp,^encoder2/fcn2/mean/BiasAdd_1/ReadVariableOp)^encoder2/fcn2/mean/MatMul/ReadVariableOp+^encoder2/fcn2/mean/MatMul_1/ReadVariableOp%^encoder2/fcn3/BiasAdd/ReadVariableOp$^encoder2/fcn3/MatMul/ReadVariableOp*
T0*
_output_shapes
: 2

Identity_1Р

Identity_2Identityencoder1/fcn3/BiasAdd:output:0%^decoder1/fcn1/BiasAdd/ReadVariableOp'^decoder1/fcn1/BiasAdd_1/ReadVariableOp$^decoder1/fcn1/MatMul/ReadVariableOp&^decoder1/fcn1/MatMul_1/ReadVariableOp%^decoder1/fcn2/BiasAdd/ReadVariableOp'^decoder1/fcn2/BiasAdd_1/ReadVariableOp$^decoder1/fcn2/MatMul/ReadVariableOp&^decoder1/fcn2/MatMul_1/ReadVariableOp+^decoder1/layer_gene/BiasAdd/ReadVariableOp-^decoder1/layer_gene/BiasAdd_1/ReadVariableOp*^decoder1/layer_gene/MatMul/ReadVariableOp,^decoder1/layer_gene/MatMul_1/ReadVariableOp%^decoder2/fcn1/BiasAdd/ReadVariableOp'^decoder2/fcn1/BiasAdd_1/ReadVariableOp$^decoder2/fcn1/MatMul/ReadVariableOp&^decoder2/fcn1/MatMul_1/ReadVariableOp%^decoder2/fcn2/BiasAdd/ReadVariableOp'^decoder2/fcn2/BiasAdd_1/ReadVariableOp$^decoder2/fcn2/MatMul/ReadVariableOp&^decoder2/fcn2/MatMul_1/ReadVariableOp%^encoder1/fcn1/BiasAdd/ReadVariableOp'^encoder1/fcn1/BiasAdd_1/ReadVariableOp$^encoder1/fcn1/MatMul/ReadVariableOp&^encoder1/fcn1/MatMul_1/ReadVariableOp*^encoder1/fcn2/mean/BiasAdd/ReadVariableOp,^encoder1/fcn2/mean/BiasAdd_1/ReadVariableOp)^encoder1/fcn2/mean/MatMul/ReadVariableOp+^encoder1/fcn2/mean/MatMul_1/ReadVariableOp%^encoder1/fcn3/BiasAdd/ReadVariableOp$^encoder1/fcn3/MatMul/ReadVariableOp*^encoder1/fcn3_gene/BiasAdd/ReadVariableOp,^encoder1/fcn3_gene/BiasAdd_1/ReadVariableOp)^encoder1/fcn3_gene/MatMul/ReadVariableOp+^encoder1/fcn3_gene/MatMul_1/ReadVariableOp%^encoder2/fcn1/BiasAdd/ReadVariableOp'^encoder2/fcn1/BiasAdd_1/ReadVariableOp$^encoder2/fcn1/MatMul/ReadVariableOp&^encoder2/fcn1/MatMul_1/ReadVariableOp*^encoder2/fcn2/mean/BiasAdd/ReadVariableOp,^encoder2/fcn2/mean/BiasAdd_1/ReadVariableOp)^encoder2/fcn2/mean/MatMul/ReadVariableOp+^encoder2/fcn2/mean/MatMul_1/ReadVariableOp%^encoder2/fcn3/BiasAdd/ReadVariableOp$^encoder2/fcn3/MatMul/ReadVariableOp*
T0*'
_output_shapes
:         P2

Identity_2Р

Identity_3Identityencoder2/fcn3/BiasAdd:output:0%^decoder1/fcn1/BiasAdd/ReadVariableOp'^decoder1/fcn1/BiasAdd_1/ReadVariableOp$^decoder1/fcn1/MatMul/ReadVariableOp&^decoder1/fcn1/MatMul_1/ReadVariableOp%^decoder1/fcn2/BiasAdd/ReadVariableOp'^decoder1/fcn2/BiasAdd_1/ReadVariableOp$^decoder1/fcn2/MatMul/ReadVariableOp&^decoder1/fcn2/MatMul_1/ReadVariableOp+^decoder1/layer_gene/BiasAdd/ReadVariableOp-^decoder1/layer_gene/BiasAdd_1/ReadVariableOp*^decoder1/layer_gene/MatMul/ReadVariableOp,^decoder1/layer_gene/MatMul_1/ReadVariableOp%^decoder2/fcn1/BiasAdd/ReadVariableOp'^decoder2/fcn1/BiasAdd_1/ReadVariableOp$^decoder2/fcn1/MatMul/ReadVariableOp&^decoder2/fcn1/MatMul_1/ReadVariableOp%^decoder2/fcn2/BiasAdd/ReadVariableOp'^decoder2/fcn2/BiasAdd_1/ReadVariableOp$^decoder2/fcn2/MatMul/ReadVariableOp&^decoder2/fcn2/MatMul_1/ReadVariableOp%^encoder1/fcn1/BiasAdd/ReadVariableOp'^encoder1/fcn1/BiasAdd_1/ReadVariableOp$^encoder1/fcn1/MatMul/ReadVariableOp&^encoder1/fcn1/MatMul_1/ReadVariableOp*^encoder1/fcn2/mean/BiasAdd/ReadVariableOp,^encoder1/fcn2/mean/BiasAdd_1/ReadVariableOp)^encoder1/fcn2/mean/MatMul/ReadVariableOp+^encoder1/fcn2/mean/MatMul_1/ReadVariableOp%^encoder1/fcn3/BiasAdd/ReadVariableOp$^encoder1/fcn3/MatMul/ReadVariableOp*^encoder1/fcn3_gene/BiasAdd/ReadVariableOp,^encoder1/fcn3_gene/BiasAdd_1/ReadVariableOp)^encoder1/fcn3_gene/MatMul/ReadVariableOp+^encoder1/fcn3_gene/MatMul_1/ReadVariableOp%^encoder2/fcn1/BiasAdd/ReadVariableOp'^encoder2/fcn1/BiasAdd_1/ReadVariableOp$^encoder2/fcn1/MatMul/ReadVariableOp&^encoder2/fcn1/MatMul_1/ReadVariableOp*^encoder2/fcn2/mean/BiasAdd/ReadVariableOp,^encoder2/fcn2/mean/BiasAdd_1/ReadVariableOp)^encoder2/fcn2/mean/MatMul/ReadVariableOp+^encoder2/fcn2/mean/MatMul_1/ReadVariableOp%^encoder2/fcn3/BiasAdd/ReadVariableOp$^encoder2/fcn3/MatMul/ReadVariableOp*
T0*'
_output_shapes
:         P2

Identity_3"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0*(
_construction_contextkEagerRuntime*┐
_input_shapesн
к:         А:         А:         А:         А:         А:         А: : : : : : : : : : : : : : : : : : : : : : : : : 2L
$decoder1/fcn1/BiasAdd/ReadVariableOp$decoder1/fcn1/BiasAdd/ReadVariableOp2P
&decoder1/fcn1/BiasAdd_1/ReadVariableOp&decoder1/fcn1/BiasAdd_1/ReadVariableOp2J
#decoder1/fcn1/MatMul/ReadVariableOp#decoder1/fcn1/MatMul/ReadVariableOp2N
%decoder1/fcn1/MatMul_1/ReadVariableOp%decoder1/fcn1/MatMul_1/ReadVariableOp2L
$decoder1/fcn2/BiasAdd/ReadVariableOp$decoder1/fcn2/BiasAdd/ReadVariableOp2P
&decoder1/fcn2/BiasAdd_1/ReadVariableOp&decoder1/fcn2/BiasAdd_1/ReadVariableOp2J
#decoder1/fcn2/MatMul/ReadVariableOp#decoder1/fcn2/MatMul/ReadVariableOp2N
%decoder1/fcn2/MatMul_1/ReadVariableOp%decoder1/fcn2/MatMul_1/ReadVariableOp2X
*decoder1/layer_gene/BiasAdd/ReadVariableOp*decoder1/layer_gene/BiasAdd/ReadVariableOp2\
,decoder1/layer_gene/BiasAdd_1/ReadVariableOp,decoder1/layer_gene/BiasAdd_1/ReadVariableOp2V
)decoder1/layer_gene/MatMul/ReadVariableOp)decoder1/layer_gene/MatMul/ReadVariableOp2Z
+decoder1/layer_gene/MatMul_1/ReadVariableOp+decoder1/layer_gene/MatMul_1/ReadVariableOp2L
$decoder2/fcn1/BiasAdd/ReadVariableOp$decoder2/fcn1/BiasAdd/ReadVariableOp2P
&decoder2/fcn1/BiasAdd_1/ReadVariableOp&decoder2/fcn1/BiasAdd_1/ReadVariableOp2J
#decoder2/fcn1/MatMul/ReadVariableOp#decoder2/fcn1/MatMul/ReadVariableOp2N
%decoder2/fcn1/MatMul_1/ReadVariableOp%decoder2/fcn1/MatMul_1/ReadVariableOp2L
$decoder2/fcn2/BiasAdd/ReadVariableOp$decoder2/fcn2/BiasAdd/ReadVariableOp2P
&decoder2/fcn2/BiasAdd_1/ReadVariableOp&decoder2/fcn2/BiasAdd_1/ReadVariableOp2J
#decoder2/fcn2/MatMul/ReadVariableOp#decoder2/fcn2/MatMul/ReadVariableOp2N
%decoder2/fcn2/MatMul_1/ReadVariableOp%decoder2/fcn2/MatMul_1/ReadVariableOp2L
$encoder1/fcn1/BiasAdd/ReadVariableOp$encoder1/fcn1/BiasAdd/ReadVariableOp2P
&encoder1/fcn1/BiasAdd_1/ReadVariableOp&encoder1/fcn1/BiasAdd_1/ReadVariableOp2J
#encoder1/fcn1/MatMul/ReadVariableOp#encoder1/fcn1/MatMul/ReadVariableOp2N
%encoder1/fcn1/MatMul_1/ReadVariableOp%encoder1/fcn1/MatMul_1/ReadVariableOp2V
)encoder1/fcn2/mean/BiasAdd/ReadVariableOp)encoder1/fcn2/mean/BiasAdd/ReadVariableOp2Z
+encoder1/fcn2/mean/BiasAdd_1/ReadVariableOp+encoder1/fcn2/mean/BiasAdd_1/ReadVariableOp2T
(encoder1/fcn2/mean/MatMul/ReadVariableOp(encoder1/fcn2/mean/MatMul/ReadVariableOp2X
*encoder1/fcn2/mean/MatMul_1/ReadVariableOp*encoder1/fcn2/mean/MatMul_1/ReadVariableOp2L
$encoder1/fcn3/BiasAdd/ReadVariableOp$encoder1/fcn3/BiasAdd/ReadVariableOp2J
#encoder1/fcn3/MatMul/ReadVariableOp#encoder1/fcn3/MatMul/ReadVariableOp2V
)encoder1/fcn3_gene/BiasAdd/ReadVariableOp)encoder1/fcn3_gene/BiasAdd/ReadVariableOp2Z
+encoder1/fcn3_gene/BiasAdd_1/ReadVariableOp+encoder1/fcn3_gene/BiasAdd_1/ReadVariableOp2T
(encoder1/fcn3_gene/MatMul/ReadVariableOp(encoder1/fcn3_gene/MatMul/ReadVariableOp2X
*encoder1/fcn3_gene/MatMul_1/ReadVariableOp*encoder1/fcn3_gene/MatMul_1/ReadVariableOp2L
$encoder2/fcn1/BiasAdd/ReadVariableOp$encoder2/fcn1/BiasAdd/ReadVariableOp2P
&encoder2/fcn1/BiasAdd_1/ReadVariableOp&encoder2/fcn1/BiasAdd_1/ReadVariableOp2J
#encoder2/fcn1/MatMul/ReadVariableOp#encoder2/fcn1/MatMul/ReadVariableOp2N
%encoder2/fcn1/MatMul_1/ReadVariableOp%encoder2/fcn1/MatMul_1/ReadVariableOp2V
)encoder2/fcn2/mean/BiasAdd/ReadVariableOp)encoder2/fcn2/mean/BiasAdd/ReadVariableOp2Z
+encoder2/fcn2/mean/BiasAdd_1/ReadVariableOp+encoder2/fcn2/mean/BiasAdd_1/ReadVariableOp2T
(encoder2/fcn2/mean/MatMul/ReadVariableOp(encoder2/fcn2/mean/MatMul/ReadVariableOp2X
*encoder2/fcn2/mean/MatMul_1/ReadVariableOp*encoder2/fcn2/mean/MatMul_1/ReadVariableOp2L
$encoder2/fcn3/BiasAdd/ReadVariableOp$encoder2/fcn3/BiasAdd/ReadVariableOp2J
#encoder2/fcn3/MatMul/ReadVariableOp#encoder2/fcn3/MatMul/ReadVariableOp:R N
(
_output_shapes
:         А
"
_user_specified_name
inputs/0:RN
(
_output_shapes
:         А
"
_user_specified_name
inputs/1:RN
(
_output_shapes
:         А
"
_user_specified_name
inputs/2:RN
(
_output_shapes
:         А
"
_user_specified_name
inputs/3:RN
(
_output_shapes
:         А
"
_user_specified_name
inputs/4:RN
(
_output_shapes
:         А
"
_user_specified_name
inputs/5:

_output_shapes
: 
п
Ю
0__inference_encoder1/fcn1_layer_call_fn_40051435

inputs
unknown:	Аd
	unknown_0:d
identityИвStatefulPartitionedCall■
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         d*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *T
fORM
K__inference_encoder1/fcn1_layer_call_and_return_conditional_losses_400495182
StatefulPartitionedCallО
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:         d2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:         А: : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:         А
 
_user_specified_nameinputs
м
Э
0__inference_encoder1/fcn3_layer_call_fn_40051611

inputs
unknown:dP
	unknown_0:P
identityИвStatefulPartitionedCall■
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         P*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *T
fORM
K__inference_encoder1/fcn3_layer_call_and_return_conditional_losses_400497342
StatefulPartitionedCallО
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:         P2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:         d: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:         d
 
_user_specified_nameinputs
╞

■
K__inference_decoder1/fcn1_layer_call_and_return_conditional_losses_40049630

inputs1
matmul_readvariableop_resource:	dА	.
biasadd_readvariableop_resource:	А	
identityИвBiasAdd/ReadVariableOpвMatMul/ReadVariableOpО
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	dА	*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         А	2
MatMulН
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:А	*
dtype02
BiasAdd/ReadVariableOpВ
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         А	2	
BiasAdd_
	LeakyRelu	LeakyReluBiasAdd:output:0*(
_output_shapes
:         А	2
	LeakyReluЭ
IdentityIdentityLeakyRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*(
_output_shapes
:         А	2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:         d: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:         d
 
_user_specified_nameinputs
╡Н
К
C__inference_model_layer_call_and_return_conditional_losses_40050226

inputs
inputs_1
inputs_2
inputs_3
inputs_4
inputs_5)
encoder2_fcn1_40050041:	Аx$
encoder2_fcn1_40050043:x-
encoder2_fcn2_mean_40050046:x`)
encoder2_fcn2_mean_40050048:`.
encoder1_fcn3_gene_40050051:	Аd)
encoder1_fcn3_gene_40050053:d)
encoder1_fcn1_40050056:	Аd$
encoder1_fcn1_40050058:d-
encoder1_fcn2_mean_40050081:dd)
encoder1_fcn2_mean_40050083:d.
decoder1_layer_gene_40050096:dd*
decoder1_layer_gene_40050098:d)
decoder2_fcn1_40050115:	`А	%
decoder2_fcn1_40050117:	А	)
decoder1_fcn1_40050120:	dА	%
decoder1_fcn1_40050122:	А	*
decoder2_fcn2_40050135:
А	А%
decoder2_fcn2_40050137:	А*
decoder1_fcn2_40050142:
А	А%
decoder1_fcn2_40050144:	А(
encoder2_fcn3_40050175:`P$
encoder2_fcn3_40050177:P(
encoder1_fcn3_40050180:dP$
encoder1_fcn3_40050182:P
unknown
identity

identity_1

identity_2

identity_3Ив%decoder1/fcn1/StatefulPartitionedCallв'decoder1/fcn1/StatefulPartitionedCall_1в%decoder1/fcn2/StatefulPartitionedCallв'decoder1/fcn2/StatefulPartitionedCall_1в+decoder1/layer_gene/StatefulPartitionedCallв-decoder1/layer_gene/StatefulPartitionedCall_1в%decoder2/fcn1/StatefulPartitionedCallв'decoder2/fcn1/StatefulPartitionedCall_1в%decoder2/fcn2/StatefulPartitionedCallв'decoder2/fcn2/StatefulPartitionedCall_1в%encoder1/fcn1/StatefulPartitionedCallв'encoder1/fcn1/StatefulPartitionedCall_1в*encoder1/fcn2/mean/StatefulPartitionedCallв,encoder1/fcn2/mean/StatefulPartitionedCall_1в%encoder1/fcn3/StatefulPartitionedCallв*encoder1/fcn3_gene/StatefulPartitionedCallв,encoder1/fcn3_gene/StatefulPartitionedCall_1в%encoder2/fcn1/StatefulPartitionedCallв'encoder2/fcn1/StatefulPartitionedCall_1в*encoder2/fcn2/mean/StatefulPartitionedCallв,encoder2/fcn2/mean/StatefulPartitionedCall_1в%encoder2/fcn3/StatefulPartitionedCall╕
%encoder2/fcn1/StatefulPartitionedCallStatefulPartitionedCallinputs_1encoder2_fcn1_40050041encoder2_fcn1_40050043*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         x*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *T
fORM
K__inference_encoder2/fcn1_layer_call_and_return_conditional_losses_400494682'
%encoder2/fcn1/StatefulPartitionedCallў
*encoder2/fcn2/mean/StatefulPartitionedCallStatefulPartitionedCall.encoder2/fcn1/StatefulPartitionedCall:output:0encoder2_fcn2_mean_40050046encoder2_fcn2_mean_40050048*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         `*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *Y
fTRR
P__inference_encoder2/fcn2/mean_layer_call_and_return_conditional_losses_400494852,
*encoder2/fcn2/mean/StatefulPartitionedCall╤
*encoder1/fcn3_gene/StatefulPartitionedCallStatefulPartitionedCallinputs_3encoder1_fcn3_gene_40050051encoder1_fcn3_gene_40050053*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         d*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *Y
fTRR
P__inference_encoder1/fcn3_gene_layer_call_and_return_conditional_losses_400495012,
*encoder1/fcn3_gene/StatefulPartitionedCall╕
%encoder1/fcn1/StatefulPartitionedCallStatefulPartitionedCallinputs_2encoder1_fcn1_40050056encoder1_fcn1_40050058*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         d*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *T
fORM
K__inference_encoder1/fcn1_layer_call_and_return_conditional_losses_400495182'
%encoder1/fcn1/StatefulPartitionedCall╒
,encoder1/fcn3_gene/StatefulPartitionedCall_1StatefulPartitionedCallinputs_5encoder1_fcn3_gene_40050051encoder1_fcn3_gene_40050053*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         d*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *Y
fTRR
P__inference_encoder1/fcn3_gene_layer_call_and_return_conditional_losses_400495012.
,encoder1/fcn3_gene/StatefulPartitionedCall_1║
'encoder1/fcn1/StatefulPartitionedCall_1StatefulPartitionedCallinputsencoder1_fcn1_40050056encoder1_fcn1_40050058*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         d*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *T
fORM
K__inference_encoder1/fcn1_layer_call_and_return_conditional_losses_400495182)
'encoder1/fcn1/StatefulPartitionedCall_1г
tf.math.square_5/SquareSquare3encoder2/fcn2/mean/StatefulPartitionedCall:output:0*
T0*'
_output_shapes
:         `2
tf.math.square_5/Squareг
tf.math.square_4/SquareSquare3encoder2/fcn2/mean/StatefulPartitionedCall:output:0*
T0*'
_output_shapes
:         `2
tf.math.square_4/Squarey
tf.math.square_3/SquareSquareinputs_1*
T0*(
_output_shapes
:         А2
tf.math.square_3/Squarey
tf.math.square_2/SquareSquareinputs_1*
T0*(
_output_shapes
:         А2
tf.math.square_2/Square╬
tf.math.subtract_3/SubSub.encoder1/fcn1/StatefulPartitionedCall:output:03encoder1/fcn3_gene/StatefulPartitionedCall:output:0*
T0*'
_output_shapes
:         d2
tf.math.subtract_3/Sub╬
tf.math.subtract/SubSub0encoder1/fcn1/StatefulPartitionedCall_1:output:05encoder1/fcn3_gene/StatefulPartitionedCall_1:output:0*
T0*'
_output_shapes
:         d2
tf.math.subtract/Subг
*tf.math.reduce_sum_3/Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
valueB :
         2,
*tf.math.reduce_sum_3/Sum/reduction_indices╨
tf.math.reduce_sum_3/SumSumtf.math.square_5/Square:y:03tf.math.reduce_sum_3/Sum/reduction_indices:output:0*
T0*'
_output_shapes
:         *
	keep_dims(2
tf.math.reduce_sum_3/Sumг
*tf.math.reduce_sum_2/Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
valueB :
         2,
*tf.math.reduce_sum_2/Sum/reduction_indices╨
tf.math.reduce_sum_2/SumSumtf.math.square_4/Square:y:03tf.math.reduce_sum_2/Sum/reduction_indices:output:0*
T0*'
_output_shapes
:         *
	keep_dims(2
tf.math.reduce_sum_2/Sumг
*tf.math.reduce_sum_1/Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
valueB :
         2,
*tf.math.reduce_sum_1/Sum/reduction_indices╨
tf.math.reduce_sum_1/SumSumtf.math.square_3/Square:y:03tf.math.reduce_sum_1/Sum/reduction_indices:output:0*
T0*'
_output_shapes
:         *
	keep_dims(2
tf.math.reduce_sum_1/SumЯ
(tf.math.reduce_sum/Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
valueB :
         2*
(tf.math.reduce_sum/Sum/reduction_indices╩
tf.math.reduce_sum/SumSumtf.math.square_2/Square:y:01tf.math.reduce_sum/Sum/reduction_indices:output:0*
T0*'
_output_shapes
:         *
	keep_dims(2
tf.math.reduce_sum/Sumу
*encoder1/fcn2/mean/StatefulPartitionedCallStatefulPartitionedCalltf.math.subtract_3/Sub:z:0encoder1_fcn2_mean_40050081encoder1_fcn2_mean_40050083*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         d*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *Y
fTRR
P__inference_encoder1/fcn2/mean_layer_call_and_return_conditional_losses_400495552,
*encoder1/fcn2/mean/StatefulPartitionedCallх
,encoder1/fcn2/mean/StatefulPartitionedCall_1StatefulPartitionedCalltf.math.subtract/Sub:z:0encoder1_fcn2_mean_40050081encoder1_fcn2_mean_40050083*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         d*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *Y
fTRR
P__inference_encoder1/fcn2/mean_layer_call_and_return_conditional_losses_400495552.
,encoder1/fcn2/mean/StatefulPartitionedCall_1З
tf.math.sqrt_2/SqrtSqrt!tf.math.reduce_sum_2/Sum:output:0*
T0*'
_output_shapes
:         2
tf.math.sqrt_2/SqrtЗ
tf.math.sqrt_3/SqrtSqrt!tf.math.reduce_sum_3/Sum:output:0*
T0*'
_output_shapes
:         2
tf.math.sqrt_3/SqrtБ
tf.math.sqrt/SqrtSqrttf.math.reduce_sum/Sum:output:0*
T0*'
_output_shapes
:         2
tf.math.sqrt/SqrtЗ
tf.math.sqrt_1/SqrtSqrt!tf.math.reduce_sum_1/Sum:output:0*
T0*'
_output_shapes
:         2
tf.math.sqrt_1/Sqrt╝
'encoder2/fcn1/StatefulPartitionedCall_1StatefulPartitionedCallinputs_4encoder2_fcn1_40050041encoder2_fcn1_40050043*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         x*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *T
fORM
K__inference_encoder2/fcn1_layer_call_and_return_conditional_losses_400494682)
'encoder2/fcn1/StatefulPartitionedCall_1Б
+decoder1/layer_gene/StatefulPartitionedCallStatefulPartitionedCall3encoder1/fcn2/mean/StatefulPartitionedCall:output:0decoder1_layer_gene_40050096decoder1_layer_gene_40050098*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         d*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *Z
fURS
Q__inference_decoder1/layer_gene_layer_call_and_return_conditional_losses_400495822-
+decoder1/layer_gene/StatefulPartitionedCallЗ
-decoder1/layer_gene/StatefulPartitionedCall_1StatefulPartitionedCall5encoder1/fcn2/mean/StatefulPartitionedCall_1:output:0decoder1_layer_gene_40050096decoder1_layer_gene_40050098*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         d*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *Z
fURS
Q__inference_decoder1/layer_gene_layer_call_and_return_conditional_losses_400495822/
-decoder1/layer_gene/StatefulPartitionedCall_1°
tf.linalg.matmul_1/MatMulMatMul3encoder2/fcn2/mean/StatefulPartitionedCall:output:03encoder2/fcn2/mean/StatefulPartitionedCall:output:0*
T0*0
_output_shapes
:                  *
transpose_b(2
tf.linalg.matmul_1/MatMulЫ
tf.math.multiply_3/MulMultf.math.sqrt_2/Sqrt:y:0tf.math.sqrt_3/Sqrt:y:0*
T0*'
_output_shapes
:         2
tf.math.multiply_3/MulЮ
tf.linalg.matmul/MatMulMatMulinputs_1inputs_1*
T0*0
_output_shapes
:                  *
transpose_b(2
tf.linalg.matmul/MatMulЩ
tf.math.multiply_2/MulMultf.math.sqrt/Sqrt:y:0tf.math.sqrt_1/Sqrt:y:0*
T0*'
_output_shapes
:         2
tf.math.multiply_2/Mul¤
,encoder2/fcn2/mean/StatefulPartitionedCall_1StatefulPartitionedCall0encoder2/fcn1/StatefulPartitionedCall_1:output:0encoder2_fcn2_mean_40050046encoder2_fcn2_mean_40050048*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         `*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *Y
fTRR
P__inference_encoder2/fcn2/mean_layer_call_and_return_conditional_losses_400494852.
,encoder2/fcn2/mean/StatefulPartitionedCall_1т
tf.__operators__.add_2/AddV2AddV24decoder1/layer_gene/StatefulPartitionedCall:output:03encoder1/fcn3_gene/StatefulPartitionedCall:output:0*
T0*'
_output_shapes
:         d2
tf.__operators__.add_2/AddV2т
tf.__operators__.add/AddV2AddV26decoder1/layer_gene/StatefulPartitionedCall_1:output:05encoder1/fcn3_gene/StatefulPartitionedCall_1:output:0*
T0*'
_output_shapes
:         d2
tf.__operators__.add/AddV2╜
tf.math.truediv_1/truedivRealDiv#tf.linalg.matmul_1/MatMul:product:0tf.math.multiply_3/Mul:z:0*
T0*0
_output_shapes
:                  2
tf.math.truediv_1/truediv╖
tf.math.truediv/truedivRealDiv!tf.linalg.matmul/MatMul:product:0tf.math.multiply_2/Mul:z:0*
T0*0
_output_shapes
:                  2
tf.math.truediv/truedivц
%decoder2/fcn1/StatefulPartitionedCallStatefulPartitionedCall5encoder2/fcn2/mean/StatefulPartitionedCall_1:output:0decoder2_fcn1_40050115decoder2_fcn1_40050117*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         А	*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *T
fORM
K__inference_decoder2/fcn1_layer_call_and_return_conditional_losses_400496132'
%decoder2/fcn1/StatefulPartitionedCall╤
%decoder1/fcn1/StatefulPartitionedCallStatefulPartitionedCall tf.__operators__.add_2/AddV2:z:0decoder1_fcn1_40050120decoder1_fcn1_40050122*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         А	*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *T
fORM
K__inference_decoder1/fcn1_layer_call_and_return_conditional_losses_400496302'
%decoder1/fcn1/StatefulPartitionedCallш
'decoder2/fcn1/StatefulPartitionedCall_1StatefulPartitionedCall3encoder2/fcn2/mean/StatefulPartitionedCall:output:0decoder2_fcn1_40050115decoder2_fcn1_40050117*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         А	*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *T
fORM
K__inference_decoder2/fcn1_layer_call_and_return_conditional_losses_400496132)
'decoder2/fcn1/StatefulPartitionedCall_1╙
'decoder1/fcn1/StatefulPartitionedCall_1StatefulPartitionedCalltf.__operators__.add/AddV2:z:0decoder1_fcn1_40050120decoder1_fcn1_40050122*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         А	*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *T
fORM
K__inference_decoder1/fcn1_layer_call_and_return_conditional_losses_400496302)
'decoder1/fcn1/StatefulPartitionedCall_1Ч
tf.nn.softmax_1/SoftmaxSoftmaxtf.math.truediv_1/truediv:z:0*
T0*0
_output_shapes
:                  2
tf.nn.softmax_1/SoftmaxС
tf.nn.softmax/SoftmaxSoftmaxtf.math.truediv/truediv:z:0*
T0*0
_output_shapes
:                  2
tf.nn.softmax/Softmaxy
tf.math.multiply_9/Mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *  HB2
tf.math.multiply_9/Mul/yЧ
tf.math.multiply_9/MulMulinputs_4!tf.math.multiply_9/Mul/y:output:0*
T0*(
_output_shapes
:         А2
tf.math.multiply_9/Mul▀
%decoder2/fcn2/StatefulPartitionedCallStatefulPartitionedCall.decoder2/fcn1/StatefulPartitionedCall:output:0decoder2_fcn2_40050135decoder2_fcn2_40050137*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         А*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *T
fORM
K__inference_decoder2/fcn2_layer_call_and_return_conditional_losses_400496562'
%decoder2/fcn2/StatefulPartitionedCally
tf.math.multiply_8/Mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *  HB2
tf.math.multiply_8/Mul/yЧ
tf.math.multiply_8/MulMulinputs_2!tf.math.multiply_8/Mul/y:output:0*
T0*(
_output_shapes
:         А2
tf.math.multiply_8/Mul▀
%decoder1/fcn2/StatefulPartitionedCallStatefulPartitionedCall.decoder1/fcn1/StatefulPartitionedCall:output:0decoder1_fcn2_40050142decoder1_fcn2_40050144*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         А*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *T
fORM
K__inference_decoder1/fcn2_layer_call_and_return_conditional_losses_400496742'
%decoder1/fcn2/StatefulPartitionedCallх
'decoder2/fcn2/StatefulPartitionedCall_1StatefulPartitionedCall0decoder2/fcn1/StatefulPartitionedCall_1:output:0decoder2_fcn2_40050135decoder2_fcn2_40050137*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         А*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *T
fORM
K__inference_decoder2/fcn2_layer_call_and_return_conditional_losses_400496562)
'decoder2/fcn2/StatefulPartitionedCall_1х
'decoder1/fcn2/StatefulPartitionedCall_1StatefulPartitionedCall0decoder1/fcn1/StatefulPartitionedCall_1:output:0decoder1_fcn2_40050142decoder1_fcn2_40050144*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         А*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *T
fORM
K__inference_decoder1/fcn2_layer_call_and_return_conditional_losses_400496742)
'decoder1/fcn2/StatefulPartitionedCall_1└
tf.math.truediv_3/truedivRealDiv!tf.nn.softmax_1/Softmax:softmax:0tf.nn.softmax/Softmax:softmax:0*
T0*0
_output_shapes
:                  2
tf.math.truediv_3/truediv└
tf.math.truediv_2/truedivRealDivtf.nn.softmax/Softmax:softmax:0!tf.nn.softmax_1/Softmax:softmax:0*
T0*0
_output_shapes
:                  2
tf.math.truediv_2/truediv╢
tf.math.subtract_5/SubSubtf.math.multiply_9/Mul:z:0.decoder2/fcn2/StatefulPartitionedCall:output:0*
T0*(
_output_shapes
:         А2
tf.math.subtract_5/Sub╢
tf.math.subtract_4/SubSubtf.math.multiply_8/Mul:z:0.decoder1/fcn2/StatefulPartitionedCall:output:0*
T0*(
_output_shapes
:         А2
tf.math.subtract_4/Suby
tf.math.multiply_1/Mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *  HB2
tf.math.multiply_1/Mul/yЧ
tf.math.multiply_1/MulMulinputs_1!tf.math.multiply_1/Mul/y:output:0*
T0*(
_output_shapes
:         А2
tf.math.multiply_1/Mulu
tf.math.multiply/Mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *  HB2
tf.math.multiply/Mul/yП
tf.math.multiply/MulMulinputstf.math.multiply/Mul/y:output:0*
T0*(
_output_shapes
:         А2
tf.math.multiply/MulЗ
tf.math.log_1/LogLogtf.math.truediv_3/truediv:z:0*
T0*0
_output_shapes
:                  2
tf.math.log_1/LogГ
tf.math.log/LogLogtf.math.truediv_2/truediv:z:0*
T0*0
_output_shapes
:                  2
tf.math.log/LogЛ
tf.math.square_9/SquareSquaretf.math.subtract_5/Sub:z:0*
T0*(
_output_shapes
:         А2
tf.math.square_9/SquareЛ
tf.math.square_8/SquareSquaretf.math.subtract_4/Sub:z:0*
T0*(
_output_shapes
:         А2
tf.math.square_8/Square╕
tf.math.subtract_2/SubSubtf.math.multiply_1/Mul:z:00decoder2/fcn2/StatefulPartitionedCall_1:output:0*
T0*(
_output_shapes
:         А2
tf.math.subtract_2/Sub╢
tf.math.subtract_1/SubSubtf.math.multiply/Mul:z:00decoder1/fcn2/StatefulPartitionedCall_1:output:0*
T0*(
_output_shapes
:         А2
tf.math.subtract_1/Subм
tf.math.multiply_5/MulMul!tf.nn.softmax_1/Softmax:softmax:0tf.math.log_1/Log:y:0*
T0*0
_output_shapes
:                  2
tf.math.multiply_5/Mulи
tf.math.multiply_4/MulMultf.nn.softmax/Softmax:softmax:0tf.math.log/Log:y:0*
T0*0
_output_shapes
:                  2
tf.math.multiply_4/MulЛ
tf.math.reduce_mean_3/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2
tf.math.reduce_mean_3/Constд
tf.math.reduce_mean_3/MeanMeantf.math.square_8/Square:y:0$tf.math.reduce_mean_3/Const:output:0*
T0*
_output_shapes
: 2
tf.math.reduce_mean_3/MeanЛ
tf.math.reduce_mean_4/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2
tf.math.reduce_mean_4/Constд
tf.math.reduce_mean_4/MeanMeantf.math.square_9/Square:y:0$tf.math.reduce_mean_4/Const:output:0*
T0*
_output_shapes
: 2
tf.math.reduce_mean_4/MeanЛ
tf.math.square_1/SquareSquaretf.math.subtract_2/Sub:z:0*
T0*(
_output_shapes
:         А2
tf.math.square_1/SquareЗ
tf.math.square/SquareSquaretf.math.subtract_1/Sub:z:0*
T0*(
_output_shapes
:         А2
tf.math.square/Squareу
%encoder2/fcn3/StatefulPartitionedCallStatefulPartitionedCall3encoder2/fcn2/mean/StatefulPartitionedCall:output:0encoder2_fcn3_40050175encoder2_fcn3_40050177*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         P*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *T
fORM
K__inference_encoder2/fcn3_layer_call_and_return_conditional_losses_400497182'
%encoder2/fcn3/StatefulPartitionedCallх
%encoder1/fcn3/StatefulPartitionedCallStatefulPartitionedCall5encoder1/fcn2/mean/StatefulPartitionedCall_1:output:0encoder1_fcn3_40050180encoder1_fcn3_40050182*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         P*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *T
fORM
K__inference_encoder1/fcn3_layer_call_and_return_conditional_losses_400497342'
%encoder1/fcn3/StatefulPartitionedCallг
*tf.math.reduce_sum_4/Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
valueB :
         2,
*tf.math.reduce_sum_4/Sum/reduction_indices║
tf.math.reduce_sum_4/SumSumtf.math.multiply_4/Mul:z:03tf.math.reduce_sum_4/Sum/reduction_indices:output:0*
T0*#
_output_shapes
:         2
tf.math.reduce_sum_4/Sumг
*tf.math.reduce_sum_5/Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
valueB :
         2,
*tf.math.reduce_sum_5/Sum/reduction_indices║
tf.math.reduce_sum_5/SumSumtf.math.multiply_5/Mul:z:03tf.math.reduce_sum_5/Sum/reduction_indices:output:0*
T0*#
_output_shapes
:         2
tf.math.reduce_sum_5/Sum▓
tf.__operators__.add_16/AddV2AddV2#tf.math.reduce_mean_3/Mean:output:0#tf.math.reduce_mean_4/Mean:output:0*
T0*
_output_shapes
: 2
tf.__operators__.add_16/AddV2З
tf.math.reduce_mean/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2
tf.math.reduce_mean/ConstЬ
tf.math.reduce_mean/MeanMeantf.math.square/Square:y:0"tf.math.reduce_mean/Const:output:0*
T0*
_output_shapes
: 2
tf.math.reduce_mean/MeanЛ
tf.math.reduce_mean_1/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2
tf.math.reduce_mean_1/Constд
tf.math.reduce_mean_1/MeanMeantf.math.square_1/Square:y:0$tf.math.reduce_mean_1/Const:output:0*
T0*
_output_shapes
: 2
tf.math.reduce_mean_1/MeanЮ
tf.math.square_7/SquareSquare.encoder2/fcn3/StatefulPartitionedCall:output:0*
T0*'
_output_shapes
:         P2
tf.math.square_7/SquareЮ
tf.math.square_6/SquareSquare.encoder1/fcn3/StatefulPartitionedCall:output:0*
T0*'
_output_shapes
:         P2
tf.math.square_6/Square╣
tf.__operators__.add_1/AddV2AddV2!tf.math.reduce_sum_4/Sum:output:0!tf.math.reduce_sum_5/Sum:output:0*
T0*#
_output_shapes
:         2
tf.__operators__.add_1/AddV2Ж
tf.math.multiply_22/MulMulunknown!tf.__operators__.add_16/AddV2:z:0*
T0*
_output_shapes
: 2
tf.math.multiply_22/Mul░
tf.__operators__.add_15/AddV2AddV2!tf.math.reduce_mean/Mean:output:0#tf.math.reduce_mean_1/Mean:output:0*
T0*
_output_shapes
: 2
tf.__operators__.add_15/AddV2г
*tf.math.reduce_sum_8/Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
valueB :
         2,
*tf.math.reduce_sum_8/Sum/reduction_indices╗
tf.math.reduce_sum_8/SumSumtf.math.square_7/Square:y:03tf.math.reduce_sum_8/Sum/reduction_indices:output:0*
T0*#
_output_shapes
:         2
tf.math.reduce_sum_8/Sumг
*tf.math.reduce_sum_7/Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
valueB :
         2,
*tf.math.reduce_sum_7/Sum/reduction_indices╗
tf.math.reduce_sum_7/SumSumtf.math.square_6/Square:y:03tf.math.reduce_sum_7/Sum/reduction_indices:output:0*
T0*#
_output_shapes
:         2
tf.math.reduce_sum_7/SumД
tf.math.reduce_mean_2/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2
tf.math.reduce_mean_2/Constй
tf.math.reduce_mean_2/MeanMean tf.__operators__.add_1/AddV2:z:0$tf.math.reduce_mean_2/Const:output:0*
T0*
_output_shapes
: 2
tf.math.reduce_mean_2/MeanБ
tf.math.truediv_21/truediv/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
tf.math.truediv_21/truediv/yо
tf.math.truediv_21/truedivRealDiv!tf.__operators__.add_15/AddV2:z:0%tf.math.truediv_21/truediv/y:output:0*
T0*
_output_shapes
: 2
tf.math.truediv_21/truedivБ
tf.math.truediv_22/truediv/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
tf.math.truediv_22/truediv/yи
tf.math.truediv_22/truedivRealDivtf.math.multiply_22/Mul:z:0%tf.math.truediv_22/truediv/y:output:0*
T0*
_output_shapes
: 2
tf.math.truediv_22/truedivГ
tf.math.sqrt_4/SqrtSqrt!tf.math.reduce_sum_7/Sum:output:0*
T0*#
_output_shapes
:         2
tf.math.sqrt_4/SqrtГ
tf.math.sqrt_5/SqrtSqrt!tf.math.reduce_sum_8/Sum:output:0*
T0*#
_output_shapes
:         2
tf.math.sqrt_5/Sqrt╔
tf.math.multiply_6/MulMul.encoder1/fcn3/StatefulPartitionedCall:output:0.encoder2/fcn3/StatefulPartitionedCall:output:0*
T0*'
_output_shapes
:         P2
tf.math.multiply_6/Mulи
tf.__operators__.add_17/AddV2AddV2tf.math.truediv_21/truediv:z:0tf.math.truediv_22/truediv:z:0*
T0*
_output_shapes
: 2
tf.__operators__.add_17/AddV2{
tf.math.multiply_23/Mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2
tf.math.multiply_23/Mul/yг
tf.math.multiply_23/MulMul#tf.math.reduce_mean_2/Mean:output:0"tf.math.multiply_23/Mul/y:output:0*
T0*
_output_shapes
: 2
tf.math.multiply_23/Mulг
*tf.math.reduce_sum_6/Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
valueB :
         2,
*tf.math.reduce_sum_6/Sum/reduction_indices║
tf.math.reduce_sum_6/SumSumtf.math.multiply_6/Mul:z:03tf.math.reduce_sum_6/Sum/reduction_indices:output:0*
T0*#
_output_shapes
:         2
tf.math.reduce_sum_6/SumЧ
tf.math.multiply_7/MulMultf.math.sqrt_4/Sqrt:y:0tf.math.sqrt_5/Sqrt:y:0*
T0*#
_output_shapes
:         2
tf.math.multiply_7/Mulи
tf.__operators__.add_18/AddV2AddV2!tf.__operators__.add_17/AddV2:z:0tf.math.multiply_23/Mul:z:0*
T0*
_output_shapes
: 2
tf.__operators__.add_18/AddV2о
tf.math.truediv_4/truedivRealDiv!tf.math.reduce_sum_6/Sum:output:0tf.math.multiply_7/Mul:z:0*
T0*#
_output_shapes
:         2
tf.math.truediv_4/truedivЫ
IdentityIdentitytf.math.truediv_4/truediv:z:0&^decoder1/fcn1/StatefulPartitionedCall(^decoder1/fcn1/StatefulPartitionedCall_1&^decoder1/fcn2/StatefulPartitionedCall(^decoder1/fcn2/StatefulPartitionedCall_1,^decoder1/layer_gene/StatefulPartitionedCall.^decoder1/layer_gene/StatefulPartitionedCall_1&^decoder2/fcn1/StatefulPartitionedCall(^decoder2/fcn1/StatefulPartitionedCall_1&^decoder2/fcn2/StatefulPartitionedCall(^decoder2/fcn2/StatefulPartitionedCall_1&^encoder1/fcn1/StatefulPartitionedCall(^encoder1/fcn1/StatefulPartitionedCall_1+^encoder1/fcn2/mean/StatefulPartitionedCall-^encoder1/fcn2/mean/StatefulPartitionedCall_1&^encoder1/fcn3/StatefulPartitionedCall+^encoder1/fcn3_gene/StatefulPartitionedCall-^encoder1/fcn3_gene/StatefulPartitionedCall_1&^encoder2/fcn1/StatefulPartitionedCall(^encoder2/fcn1/StatefulPartitionedCall_1+^encoder2/fcn2/mean/StatefulPartitionedCall-^encoder2/fcn2/mean/StatefulPartitionedCall_1&^encoder2/fcn3/StatefulPartitionedCall*
T0*#
_output_shapes
:         2

IdentityЦ

Identity_1Identity!tf.__operators__.add_18/AddV2:z:0&^decoder1/fcn1/StatefulPartitionedCall(^decoder1/fcn1/StatefulPartitionedCall_1&^decoder1/fcn2/StatefulPartitionedCall(^decoder1/fcn2/StatefulPartitionedCall_1,^decoder1/layer_gene/StatefulPartitionedCall.^decoder1/layer_gene/StatefulPartitionedCall_1&^decoder2/fcn1/StatefulPartitionedCall(^decoder2/fcn1/StatefulPartitionedCall_1&^decoder2/fcn2/StatefulPartitionedCall(^decoder2/fcn2/StatefulPartitionedCall_1&^encoder1/fcn1/StatefulPartitionedCall(^encoder1/fcn1/StatefulPartitionedCall_1+^encoder1/fcn2/mean/StatefulPartitionedCall-^encoder1/fcn2/mean/StatefulPartitionedCall_1&^encoder1/fcn3/StatefulPartitionedCall+^encoder1/fcn3_gene/StatefulPartitionedCall-^encoder1/fcn3_gene/StatefulPartitionedCall_1&^encoder2/fcn1/StatefulPartitionedCall(^encoder2/fcn1/StatefulPartitionedCall_1+^encoder2/fcn2/mean/StatefulPartitionedCall-^encoder2/fcn2/mean/StatefulPartitionedCall_1&^encoder2/fcn3/StatefulPartitionedCall*
T0*
_output_shapes
: 2

Identity_1┤

Identity_2Identity.encoder1/fcn3/StatefulPartitionedCall:output:0&^decoder1/fcn1/StatefulPartitionedCall(^decoder1/fcn1/StatefulPartitionedCall_1&^decoder1/fcn2/StatefulPartitionedCall(^decoder1/fcn2/StatefulPartitionedCall_1,^decoder1/layer_gene/StatefulPartitionedCall.^decoder1/layer_gene/StatefulPartitionedCall_1&^decoder2/fcn1/StatefulPartitionedCall(^decoder2/fcn1/StatefulPartitionedCall_1&^decoder2/fcn2/StatefulPartitionedCall(^decoder2/fcn2/StatefulPartitionedCall_1&^encoder1/fcn1/StatefulPartitionedCall(^encoder1/fcn1/StatefulPartitionedCall_1+^encoder1/fcn2/mean/StatefulPartitionedCall-^encoder1/fcn2/mean/StatefulPartitionedCall_1&^encoder1/fcn3/StatefulPartitionedCall+^encoder1/fcn3_gene/StatefulPartitionedCall-^encoder1/fcn3_gene/StatefulPartitionedCall_1&^encoder2/fcn1/StatefulPartitionedCall(^encoder2/fcn1/StatefulPartitionedCall_1+^encoder2/fcn2/mean/StatefulPartitionedCall-^encoder2/fcn2/mean/StatefulPartitionedCall_1&^encoder2/fcn3/StatefulPartitionedCall*
T0*'
_output_shapes
:         P2

Identity_2┤

Identity_3Identity.encoder2/fcn3/StatefulPartitionedCall:output:0&^decoder1/fcn1/StatefulPartitionedCall(^decoder1/fcn1/StatefulPartitionedCall_1&^decoder1/fcn2/StatefulPartitionedCall(^decoder1/fcn2/StatefulPartitionedCall_1,^decoder1/layer_gene/StatefulPartitionedCall.^decoder1/layer_gene/StatefulPartitionedCall_1&^decoder2/fcn1/StatefulPartitionedCall(^decoder2/fcn1/StatefulPartitionedCall_1&^decoder2/fcn2/StatefulPartitionedCall(^decoder2/fcn2/StatefulPartitionedCall_1&^encoder1/fcn1/StatefulPartitionedCall(^encoder1/fcn1/StatefulPartitionedCall_1+^encoder1/fcn2/mean/StatefulPartitionedCall-^encoder1/fcn2/mean/StatefulPartitionedCall_1&^encoder1/fcn3/StatefulPartitionedCall+^encoder1/fcn3_gene/StatefulPartitionedCall-^encoder1/fcn3_gene/StatefulPartitionedCall_1&^encoder2/fcn1/StatefulPartitionedCall(^encoder2/fcn1/StatefulPartitionedCall_1+^encoder2/fcn2/mean/StatefulPartitionedCall-^encoder2/fcn2/mean/StatefulPartitionedCall_1&^encoder2/fcn3/StatefulPartitionedCall*
T0*'
_output_shapes
:         P2

Identity_3"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0*(
_construction_contextkEagerRuntime*┐
_input_shapesн
к:         А:         А:         А:         А:         А:         А: : : : : : : : : : : : : : : : : : : : : : : : : 2N
%decoder1/fcn1/StatefulPartitionedCall%decoder1/fcn1/StatefulPartitionedCall2R
'decoder1/fcn1/StatefulPartitionedCall_1'decoder1/fcn1/StatefulPartitionedCall_12N
%decoder1/fcn2/StatefulPartitionedCall%decoder1/fcn2/StatefulPartitionedCall2R
'decoder1/fcn2/StatefulPartitionedCall_1'decoder1/fcn2/StatefulPartitionedCall_12Z
+decoder1/layer_gene/StatefulPartitionedCall+decoder1/layer_gene/StatefulPartitionedCall2^
-decoder1/layer_gene/StatefulPartitionedCall_1-decoder1/layer_gene/StatefulPartitionedCall_12N
%decoder2/fcn1/StatefulPartitionedCall%decoder2/fcn1/StatefulPartitionedCall2R
'decoder2/fcn1/StatefulPartitionedCall_1'decoder2/fcn1/StatefulPartitionedCall_12N
%decoder2/fcn2/StatefulPartitionedCall%decoder2/fcn2/StatefulPartitionedCall2R
'decoder2/fcn2/StatefulPartitionedCall_1'decoder2/fcn2/StatefulPartitionedCall_12N
%encoder1/fcn1/StatefulPartitionedCall%encoder1/fcn1/StatefulPartitionedCall2R
'encoder1/fcn1/StatefulPartitionedCall_1'encoder1/fcn1/StatefulPartitionedCall_12X
*encoder1/fcn2/mean/StatefulPartitionedCall*encoder1/fcn2/mean/StatefulPartitionedCall2\
,encoder1/fcn2/mean/StatefulPartitionedCall_1,encoder1/fcn2/mean/StatefulPartitionedCall_12N
%encoder1/fcn3/StatefulPartitionedCall%encoder1/fcn3/StatefulPartitionedCall2X
*encoder1/fcn3_gene/StatefulPartitionedCall*encoder1/fcn3_gene/StatefulPartitionedCall2\
,encoder1/fcn3_gene/StatefulPartitionedCall_1,encoder1/fcn3_gene/StatefulPartitionedCall_12N
%encoder2/fcn1/StatefulPartitionedCall%encoder2/fcn1/StatefulPartitionedCall2R
'encoder2/fcn1/StatefulPartitionedCall_1'encoder2/fcn1/StatefulPartitionedCall_12X
*encoder2/fcn2/mean/StatefulPartitionedCall*encoder2/fcn2/mean/StatefulPartitionedCall2\
,encoder2/fcn2/mean/StatefulPartitionedCall_1,encoder2/fcn2/mean/StatefulPartitionedCall_12N
%encoder2/fcn3/StatefulPartitionedCall%encoder2/fcn3/StatefulPartitionedCall:P L
(
_output_shapes
:         А
 
_user_specified_nameinputs:PL
(
_output_shapes
:         А
 
_user_specified_nameinputs:PL
(
_output_shapes
:         А
 
_user_specified_nameinputs:PL
(
_output_shapes
:         А
 
_user_specified_nameinputs:PL
(
_output_shapes
:         А
 
_user_specified_nameinputs:PL
(
_output_shapes
:         А
 
_user_specified_nameinputs:

_output_shapes
: 
т	
 
K__inference_decoder1/fcn2_layer_call_and_return_conditional_losses_40051564

inputs2
matmul_readvariableop_resource:
А	А.
biasadd_readvariableop_resource:	А
identityИвBiasAdd/ReadVariableOpвMatMul/ReadVariableOpП
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
А	А*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         А2
MatMulН
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:А*
dtype02
BiasAdd/ReadVariableOpВ
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         А2	
BiasAddЦ
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*(
_output_shapes
:         А2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:         А	: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:         А	
 
_user_specified_nameinputs
╣
г
5__inference_encoder1/fcn3_gene_layer_call_fn_40051454

inputs
unknown:	Аd
	unknown_0:d
identityИвStatefulPartitionedCallГ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         d*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *Y
fTRR
P__inference_encoder1/fcn3_gene_layer_call_and_return_conditional_losses_400495012
StatefulPartitionedCallО
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:         d2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:         А: : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:         А
 
_user_specified_nameinputs
╔В
Ф
C__inference_model_layer_call_and_return_conditional_losses_40051034
inputs_0
inputs_1
inputs_2
inputs_3
inputs_4
inputs_5?
,encoder2_fcn1_matmul_readvariableop_resource:	Аx;
-encoder2_fcn1_biasadd_readvariableop_resource:xC
1encoder2_fcn2_mean_matmul_readvariableop_resource:x`@
2encoder2_fcn2_mean_biasadd_readvariableop_resource:`D
1encoder1_fcn3_gene_matmul_readvariableop_resource:	Аd@
2encoder1_fcn3_gene_biasadd_readvariableop_resource:d?
,encoder1_fcn1_matmul_readvariableop_resource:	Аd;
-encoder1_fcn1_biasadd_readvariableop_resource:dC
1encoder1_fcn2_mean_matmul_readvariableop_resource:dd@
2encoder1_fcn2_mean_biasadd_readvariableop_resource:dD
2decoder1_layer_gene_matmul_readvariableop_resource:ddA
3decoder1_layer_gene_biasadd_readvariableop_resource:d?
,decoder2_fcn1_matmul_readvariableop_resource:	`А	<
-decoder2_fcn1_biasadd_readvariableop_resource:	А	?
,decoder1_fcn1_matmul_readvariableop_resource:	dА	<
-decoder1_fcn1_biasadd_readvariableop_resource:	А	@
,decoder2_fcn2_matmul_readvariableop_resource:
А	А<
-decoder2_fcn2_biasadd_readvariableop_resource:	А@
,decoder1_fcn2_matmul_readvariableop_resource:
А	А<
-decoder1_fcn2_biasadd_readvariableop_resource:	А>
,encoder2_fcn3_matmul_readvariableop_resource:`P;
-encoder2_fcn3_biasadd_readvariableop_resource:P>
,encoder1_fcn3_matmul_readvariableop_resource:dP;
-encoder1_fcn3_biasadd_readvariableop_resource:P
unknown
identity

identity_1

identity_2

identity_3Ив$decoder1/fcn1/BiasAdd/ReadVariableOpв&decoder1/fcn1/BiasAdd_1/ReadVariableOpв#decoder1/fcn1/MatMul/ReadVariableOpв%decoder1/fcn1/MatMul_1/ReadVariableOpв$decoder1/fcn2/BiasAdd/ReadVariableOpв&decoder1/fcn2/BiasAdd_1/ReadVariableOpв#decoder1/fcn2/MatMul/ReadVariableOpв%decoder1/fcn2/MatMul_1/ReadVariableOpв*decoder1/layer_gene/BiasAdd/ReadVariableOpв,decoder1/layer_gene/BiasAdd_1/ReadVariableOpв)decoder1/layer_gene/MatMul/ReadVariableOpв+decoder1/layer_gene/MatMul_1/ReadVariableOpв$decoder2/fcn1/BiasAdd/ReadVariableOpв&decoder2/fcn1/BiasAdd_1/ReadVariableOpв#decoder2/fcn1/MatMul/ReadVariableOpв%decoder2/fcn1/MatMul_1/ReadVariableOpв$decoder2/fcn2/BiasAdd/ReadVariableOpв&decoder2/fcn2/BiasAdd_1/ReadVariableOpв#decoder2/fcn2/MatMul/ReadVariableOpв%decoder2/fcn2/MatMul_1/ReadVariableOpв$encoder1/fcn1/BiasAdd/ReadVariableOpв&encoder1/fcn1/BiasAdd_1/ReadVariableOpв#encoder1/fcn1/MatMul/ReadVariableOpв%encoder1/fcn1/MatMul_1/ReadVariableOpв)encoder1/fcn2/mean/BiasAdd/ReadVariableOpв+encoder1/fcn2/mean/BiasAdd_1/ReadVariableOpв(encoder1/fcn2/mean/MatMul/ReadVariableOpв*encoder1/fcn2/mean/MatMul_1/ReadVariableOpв$encoder1/fcn3/BiasAdd/ReadVariableOpв#encoder1/fcn3/MatMul/ReadVariableOpв)encoder1/fcn3_gene/BiasAdd/ReadVariableOpв+encoder1/fcn3_gene/BiasAdd_1/ReadVariableOpв(encoder1/fcn3_gene/MatMul/ReadVariableOpв*encoder1/fcn3_gene/MatMul_1/ReadVariableOpв$encoder2/fcn1/BiasAdd/ReadVariableOpв&encoder2/fcn1/BiasAdd_1/ReadVariableOpв#encoder2/fcn1/MatMul/ReadVariableOpв%encoder2/fcn1/MatMul_1/ReadVariableOpв)encoder2/fcn2/mean/BiasAdd/ReadVariableOpв+encoder2/fcn2/mean/BiasAdd_1/ReadVariableOpв(encoder2/fcn2/mean/MatMul/ReadVariableOpв*encoder2/fcn2/mean/MatMul_1/ReadVariableOpв$encoder2/fcn3/BiasAdd/ReadVariableOpв#encoder2/fcn3/MatMul/ReadVariableOp╕
#encoder2/fcn1/MatMul/ReadVariableOpReadVariableOp,encoder2_fcn1_matmul_readvariableop_resource*
_output_shapes
:	Аx*
dtype02%
#encoder2/fcn1/MatMul/ReadVariableOpЯ
encoder2/fcn1/MatMulMatMulinputs_1+encoder2/fcn1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         x2
encoder2/fcn1/MatMul╢
$encoder2/fcn1/BiasAdd/ReadVariableOpReadVariableOp-encoder2_fcn1_biasadd_readvariableop_resource*
_output_shapes
:x*
dtype02&
$encoder2/fcn1/BiasAdd/ReadVariableOp╣
encoder2/fcn1/BiasAddBiasAddencoder2/fcn1/MatMul:product:0,encoder2/fcn1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         x2
encoder2/fcn1/BiasAddВ
encoder2/fcn1/ReluReluencoder2/fcn1/BiasAdd:output:0*
T0*'
_output_shapes
:         x2
encoder2/fcn1/Relu╞
(encoder2/fcn2/mean/MatMul/ReadVariableOpReadVariableOp1encoder2_fcn2_mean_matmul_readvariableop_resource*
_output_shapes

:x`*
dtype02*
(encoder2/fcn2/mean/MatMul/ReadVariableOp╞
encoder2/fcn2/mean/MatMulMatMul encoder2/fcn1/Relu:activations:00encoder2/fcn2/mean/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         `2
encoder2/fcn2/mean/MatMul┼
)encoder2/fcn2/mean/BiasAdd/ReadVariableOpReadVariableOp2encoder2_fcn2_mean_biasadd_readvariableop_resource*
_output_shapes
:`*
dtype02+
)encoder2/fcn2/mean/BiasAdd/ReadVariableOp═
encoder2/fcn2/mean/BiasAddBiasAdd#encoder2/fcn2/mean/MatMul:product:01encoder2/fcn2/mean/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         `2
encoder2/fcn2/mean/BiasAddЧ
encoder2/fcn2/mean/LeakyRelu	LeakyRelu#encoder2/fcn2/mean/BiasAdd:output:0*'
_output_shapes
:         `2
encoder2/fcn2/mean/LeakyRelu╟
(encoder1/fcn3_gene/MatMul/ReadVariableOpReadVariableOp1encoder1_fcn3_gene_matmul_readvariableop_resource*
_output_shapes
:	Аd*
dtype02*
(encoder1/fcn3_gene/MatMul/ReadVariableOpо
encoder1/fcn3_gene/MatMulMatMulinputs_30encoder1/fcn3_gene/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         d2
encoder1/fcn3_gene/MatMul┼
)encoder1/fcn3_gene/BiasAdd/ReadVariableOpReadVariableOp2encoder1_fcn3_gene_biasadd_readvariableop_resource*
_output_shapes
:d*
dtype02+
)encoder1/fcn3_gene/BiasAdd/ReadVariableOp═
encoder1/fcn3_gene/BiasAddBiasAdd#encoder1/fcn3_gene/MatMul:product:01encoder1/fcn3_gene/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         d2
encoder1/fcn3_gene/BiasAdd╕
#encoder1/fcn1/MatMul/ReadVariableOpReadVariableOp,encoder1_fcn1_matmul_readvariableop_resource*
_output_shapes
:	Аd*
dtype02%
#encoder1/fcn1/MatMul/ReadVariableOpЯ
encoder1/fcn1/MatMulMatMulinputs_2+encoder1/fcn1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         d2
encoder1/fcn1/MatMul╢
$encoder1/fcn1/BiasAdd/ReadVariableOpReadVariableOp-encoder1_fcn1_biasadd_readvariableop_resource*
_output_shapes
:d*
dtype02&
$encoder1/fcn1/BiasAdd/ReadVariableOp╣
encoder1/fcn1/BiasAddBiasAddencoder1/fcn1/MatMul:product:0,encoder1/fcn1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         d2
encoder1/fcn1/BiasAddВ
encoder1/fcn1/ReluReluencoder1/fcn1/BiasAdd:output:0*
T0*'
_output_shapes
:         d2
encoder1/fcn1/Relu╦
*encoder1/fcn3_gene/MatMul_1/ReadVariableOpReadVariableOp1encoder1_fcn3_gene_matmul_readvariableop_resource*
_output_shapes
:	Аd*
dtype02,
*encoder1/fcn3_gene/MatMul_1/ReadVariableOp┤
encoder1/fcn3_gene/MatMul_1MatMulinputs_52encoder1/fcn3_gene/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:         d2
encoder1/fcn3_gene/MatMul_1╔
+encoder1/fcn3_gene/BiasAdd_1/ReadVariableOpReadVariableOp2encoder1_fcn3_gene_biasadd_readvariableop_resource*
_output_shapes
:d*
dtype02-
+encoder1/fcn3_gene/BiasAdd_1/ReadVariableOp╒
encoder1/fcn3_gene/BiasAdd_1BiasAdd%encoder1/fcn3_gene/MatMul_1:product:03encoder1/fcn3_gene/BiasAdd_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:         d2
encoder1/fcn3_gene/BiasAdd_1╝
%encoder1/fcn1/MatMul_1/ReadVariableOpReadVariableOp,encoder1_fcn1_matmul_readvariableop_resource*
_output_shapes
:	Аd*
dtype02'
%encoder1/fcn1/MatMul_1/ReadVariableOpе
encoder1/fcn1/MatMul_1MatMulinputs_0-encoder1/fcn1/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:         d2
encoder1/fcn1/MatMul_1║
&encoder1/fcn1/BiasAdd_1/ReadVariableOpReadVariableOp-encoder1_fcn1_biasadd_readvariableop_resource*
_output_shapes
:d*
dtype02(
&encoder1/fcn1/BiasAdd_1/ReadVariableOp┴
encoder1/fcn1/BiasAdd_1BiasAdd encoder1/fcn1/MatMul_1:product:0.encoder1/fcn1/BiasAdd_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:         d2
encoder1/fcn1/BiasAdd_1И
encoder1/fcn1/Relu_1Relu encoder1/fcn1/BiasAdd_1:output:0*
T0*'
_output_shapes
:         d2
encoder1/fcn1/Relu_1Ъ
tf.math.square_5/SquareSquare*encoder2/fcn2/mean/LeakyRelu:activations:0*
T0*'
_output_shapes
:         `2
tf.math.square_5/SquareЪ
tf.math.square_4/SquareSquare*encoder2/fcn2/mean/LeakyRelu:activations:0*
T0*'
_output_shapes
:         `2
tf.math.square_4/Squarey
tf.math.square_3/SquareSquareinputs_1*
T0*(
_output_shapes
:         А2
tf.math.square_3/Squarey
tf.math.square_2/SquareSquareinputs_1*
T0*(
_output_shapes
:         А2
tf.math.square_2/Square░
tf.math.subtract_3/SubSub encoder1/fcn1/Relu:activations:0#encoder1/fcn3_gene/BiasAdd:output:0*
T0*'
_output_shapes
:         d2
tf.math.subtract_3/Sub░
tf.math.subtract/SubSub"encoder1/fcn1/Relu_1:activations:0%encoder1/fcn3_gene/BiasAdd_1:output:0*
T0*'
_output_shapes
:         d2
tf.math.subtract/Subг
*tf.math.reduce_sum_3/Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
valueB :
         2,
*tf.math.reduce_sum_3/Sum/reduction_indices╨
tf.math.reduce_sum_3/SumSumtf.math.square_5/Square:y:03tf.math.reduce_sum_3/Sum/reduction_indices:output:0*
T0*'
_output_shapes
:         *
	keep_dims(2
tf.math.reduce_sum_3/Sumг
*tf.math.reduce_sum_2/Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
valueB :
         2,
*tf.math.reduce_sum_2/Sum/reduction_indices╨
tf.math.reduce_sum_2/SumSumtf.math.square_4/Square:y:03tf.math.reduce_sum_2/Sum/reduction_indices:output:0*
T0*'
_output_shapes
:         *
	keep_dims(2
tf.math.reduce_sum_2/Sumг
*tf.math.reduce_sum_1/Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
valueB :
         2,
*tf.math.reduce_sum_1/Sum/reduction_indices╨
tf.math.reduce_sum_1/SumSumtf.math.square_3/Square:y:03tf.math.reduce_sum_1/Sum/reduction_indices:output:0*
T0*'
_output_shapes
:         *
	keep_dims(2
tf.math.reduce_sum_1/SumЯ
(tf.math.reduce_sum/Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
valueB :
         2*
(tf.math.reduce_sum/Sum/reduction_indices╩
tf.math.reduce_sum/SumSumtf.math.square_2/Square:y:01tf.math.reduce_sum/Sum/reduction_indices:output:0*
T0*'
_output_shapes
:         *
	keep_dims(2
tf.math.reduce_sum/Sum╞
(encoder1/fcn2/mean/MatMul/ReadVariableOpReadVariableOp1encoder1_fcn2_mean_matmul_readvariableop_resource*
_output_shapes

:dd*
dtype02*
(encoder1/fcn2/mean/MatMul/ReadVariableOp└
encoder1/fcn2/mean/MatMulMatMultf.math.subtract_3/Sub:z:00encoder1/fcn2/mean/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         d2
encoder1/fcn2/mean/MatMul┼
)encoder1/fcn2/mean/BiasAdd/ReadVariableOpReadVariableOp2encoder1_fcn2_mean_biasadd_readvariableop_resource*
_output_shapes
:d*
dtype02+
)encoder1/fcn2/mean/BiasAdd/ReadVariableOp═
encoder1/fcn2/mean/BiasAddBiasAdd#encoder1/fcn2/mean/MatMul:product:01encoder1/fcn2/mean/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         d2
encoder1/fcn2/mean/BiasAddЧ
encoder1/fcn2/mean/LeakyRelu	LeakyRelu#encoder1/fcn2/mean/BiasAdd:output:0*'
_output_shapes
:         d2
encoder1/fcn2/mean/LeakyRelu╩
*encoder1/fcn2/mean/MatMul_1/ReadVariableOpReadVariableOp1encoder1_fcn2_mean_matmul_readvariableop_resource*
_output_shapes

:dd*
dtype02,
*encoder1/fcn2/mean/MatMul_1/ReadVariableOp─
encoder1/fcn2/mean/MatMul_1MatMultf.math.subtract/Sub:z:02encoder1/fcn2/mean/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:         d2
encoder1/fcn2/mean/MatMul_1╔
+encoder1/fcn2/mean/BiasAdd_1/ReadVariableOpReadVariableOp2encoder1_fcn2_mean_biasadd_readvariableop_resource*
_output_shapes
:d*
dtype02-
+encoder1/fcn2/mean/BiasAdd_1/ReadVariableOp╒
encoder1/fcn2/mean/BiasAdd_1BiasAdd%encoder1/fcn2/mean/MatMul_1:product:03encoder1/fcn2/mean/BiasAdd_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:         d2
encoder1/fcn2/mean/BiasAdd_1Э
encoder1/fcn2/mean/LeakyRelu_1	LeakyRelu%encoder1/fcn2/mean/BiasAdd_1:output:0*'
_output_shapes
:         d2 
encoder1/fcn2/mean/LeakyRelu_1З
tf.math.sqrt_2/SqrtSqrt!tf.math.reduce_sum_2/Sum:output:0*
T0*'
_output_shapes
:         2
tf.math.sqrt_2/SqrtЗ
tf.math.sqrt_3/SqrtSqrt!tf.math.reduce_sum_3/Sum:output:0*
T0*'
_output_shapes
:         2
tf.math.sqrt_3/SqrtБ
tf.math.sqrt/SqrtSqrttf.math.reduce_sum/Sum:output:0*
T0*'
_output_shapes
:         2
tf.math.sqrt/SqrtЗ
tf.math.sqrt_1/SqrtSqrt!tf.math.reduce_sum_1/Sum:output:0*
T0*'
_output_shapes
:         2
tf.math.sqrt_1/Sqrt╝
%encoder2/fcn1/MatMul_1/ReadVariableOpReadVariableOp,encoder2_fcn1_matmul_readvariableop_resource*
_output_shapes
:	Аx*
dtype02'
%encoder2/fcn1/MatMul_1/ReadVariableOpе
encoder2/fcn1/MatMul_1MatMulinputs_4-encoder2/fcn1/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:         x2
encoder2/fcn1/MatMul_1║
&encoder2/fcn1/BiasAdd_1/ReadVariableOpReadVariableOp-encoder2_fcn1_biasadd_readvariableop_resource*
_output_shapes
:x*
dtype02(
&encoder2/fcn1/BiasAdd_1/ReadVariableOp┴
encoder2/fcn1/BiasAdd_1BiasAdd encoder2/fcn1/MatMul_1:product:0.encoder2/fcn1/BiasAdd_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:         x2
encoder2/fcn1/BiasAdd_1И
encoder2/fcn1/Relu_1Relu encoder2/fcn1/BiasAdd_1:output:0*
T0*'
_output_shapes
:         x2
encoder2/fcn1/Relu_1╔
)decoder1/layer_gene/MatMul/ReadVariableOpReadVariableOp2decoder1_layer_gene_matmul_readvariableop_resource*
_output_shapes

:dd*
dtype02+
)decoder1/layer_gene/MatMul/ReadVariableOp╙
decoder1/layer_gene/MatMulMatMul*encoder1/fcn2/mean/LeakyRelu:activations:01decoder1/layer_gene/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         d2
decoder1/layer_gene/MatMul╚
*decoder1/layer_gene/BiasAdd/ReadVariableOpReadVariableOp3decoder1_layer_gene_biasadd_readvariableop_resource*
_output_shapes
:d*
dtype02,
*decoder1/layer_gene/BiasAdd/ReadVariableOp╤
decoder1/layer_gene/BiasAddBiasAdd$decoder1/layer_gene/MatMul:product:02decoder1/layer_gene/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         d2
decoder1/layer_gene/BiasAddЪ
decoder1/layer_gene/LeakyRelu	LeakyRelu$decoder1/layer_gene/BiasAdd:output:0*'
_output_shapes
:         d2
decoder1/layer_gene/LeakyRelu═
+decoder1/layer_gene/MatMul_1/ReadVariableOpReadVariableOp2decoder1_layer_gene_matmul_readvariableop_resource*
_output_shapes

:dd*
dtype02-
+decoder1/layer_gene/MatMul_1/ReadVariableOp█
decoder1/layer_gene/MatMul_1MatMul,encoder1/fcn2/mean/LeakyRelu_1:activations:03decoder1/layer_gene/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:         d2
decoder1/layer_gene/MatMul_1╠
,decoder1/layer_gene/BiasAdd_1/ReadVariableOpReadVariableOp3decoder1_layer_gene_biasadd_readvariableop_resource*
_output_shapes
:d*
dtype02.
,decoder1/layer_gene/BiasAdd_1/ReadVariableOp┘
decoder1/layer_gene/BiasAdd_1BiasAdd&decoder1/layer_gene/MatMul_1:product:04decoder1/layer_gene/BiasAdd_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:         d2
decoder1/layer_gene/BiasAdd_1а
decoder1/layer_gene/LeakyRelu_1	LeakyRelu&decoder1/layer_gene/BiasAdd_1:output:0*'
_output_shapes
:         d2!
decoder1/layer_gene/LeakyRelu_1ц
tf.linalg.matmul_1/MatMulMatMul*encoder2/fcn2/mean/LeakyRelu:activations:0*encoder2/fcn2/mean/LeakyRelu:activations:0*
T0*0
_output_shapes
:                  *
transpose_b(2
tf.linalg.matmul_1/MatMulЫ
tf.math.multiply_3/MulMultf.math.sqrt_2/Sqrt:y:0tf.math.sqrt_3/Sqrt:y:0*
T0*'
_output_shapes
:         2
tf.math.multiply_3/MulЮ
tf.linalg.matmul/MatMulMatMulinputs_1inputs_1*
T0*0
_output_shapes
:                  *
transpose_b(2
tf.linalg.matmul/MatMulЩ
tf.math.multiply_2/MulMultf.math.sqrt/Sqrt:y:0tf.math.sqrt_1/Sqrt:y:0*
T0*'
_output_shapes
:         2
tf.math.multiply_2/Mul╩
*encoder2/fcn2/mean/MatMul_1/ReadVariableOpReadVariableOp1encoder2_fcn2_mean_matmul_readvariableop_resource*
_output_shapes

:x`*
dtype02,
*encoder2/fcn2/mean/MatMul_1/ReadVariableOp╬
encoder2/fcn2/mean/MatMul_1MatMul"encoder2/fcn1/Relu_1:activations:02encoder2/fcn2/mean/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:         `2
encoder2/fcn2/mean/MatMul_1╔
+encoder2/fcn2/mean/BiasAdd_1/ReadVariableOpReadVariableOp2encoder2_fcn2_mean_biasadd_readvariableop_resource*
_output_shapes
:`*
dtype02-
+encoder2/fcn2/mean/BiasAdd_1/ReadVariableOp╒
encoder2/fcn2/mean/BiasAdd_1BiasAdd%encoder2/fcn2/mean/MatMul_1:product:03encoder2/fcn2/mean/BiasAdd_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:         `2
encoder2/fcn2/mean/BiasAdd_1Э
encoder2/fcn2/mean/LeakyRelu_1	LeakyRelu%encoder2/fcn2/mean/BiasAdd_1:output:0*'
_output_shapes
:         `2 
encoder2/fcn2/mean/LeakyRelu_1╔
tf.__operators__.add_2/AddV2AddV2+decoder1/layer_gene/LeakyRelu:activations:0#encoder1/fcn3_gene/BiasAdd:output:0*
T0*'
_output_shapes
:         d2
tf.__operators__.add_2/AddV2╔
tf.__operators__.add/AddV2AddV2-decoder1/layer_gene/LeakyRelu_1:activations:0%encoder1/fcn3_gene/BiasAdd_1:output:0*
T0*'
_output_shapes
:         d2
tf.__operators__.add/AddV2╜
tf.math.truediv_1/truedivRealDiv#tf.linalg.matmul_1/MatMul:product:0tf.math.multiply_3/Mul:z:0*
T0*0
_output_shapes
:                  2
tf.math.truediv_1/truediv╖
tf.math.truediv/truedivRealDiv!tf.linalg.matmul/MatMul:product:0tf.math.multiply_2/Mul:z:0*
T0*0
_output_shapes
:                  2
tf.math.truediv/truediv╕
#decoder2/fcn1/MatMul/ReadVariableOpReadVariableOp,decoder2_fcn1_matmul_readvariableop_resource*
_output_shapes
:	`А	*
dtype02%
#decoder2/fcn1/MatMul/ReadVariableOp─
decoder2/fcn1/MatMulMatMul,encoder2/fcn2/mean/LeakyRelu_1:activations:0+decoder2/fcn1/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         А	2
decoder2/fcn1/MatMul╖
$decoder2/fcn1/BiasAdd/ReadVariableOpReadVariableOp-decoder2_fcn1_biasadd_readvariableop_resource*
_output_shapes	
:А	*
dtype02&
$decoder2/fcn1/BiasAdd/ReadVariableOp║
decoder2/fcn1/BiasAddBiasAdddecoder2/fcn1/MatMul:product:0,decoder2/fcn1/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         А	2
decoder2/fcn1/BiasAddЙ
decoder2/fcn1/LeakyRelu	LeakyReludecoder2/fcn1/BiasAdd:output:0*(
_output_shapes
:         А	2
decoder2/fcn1/LeakyRelu╕
#decoder1/fcn1/MatMul/ReadVariableOpReadVariableOp,decoder1_fcn1_matmul_readvariableop_resource*
_output_shapes
:	dА	*
dtype02%
#decoder1/fcn1/MatMul/ReadVariableOp╕
decoder1/fcn1/MatMulMatMul tf.__operators__.add_2/AddV2:z:0+decoder1/fcn1/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         А	2
decoder1/fcn1/MatMul╖
$decoder1/fcn1/BiasAdd/ReadVariableOpReadVariableOp-decoder1_fcn1_biasadd_readvariableop_resource*
_output_shapes	
:А	*
dtype02&
$decoder1/fcn1/BiasAdd/ReadVariableOp║
decoder1/fcn1/BiasAddBiasAdddecoder1/fcn1/MatMul:product:0,decoder1/fcn1/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         А	2
decoder1/fcn1/BiasAddЙ
decoder1/fcn1/LeakyRelu	LeakyReludecoder1/fcn1/BiasAdd:output:0*(
_output_shapes
:         А	2
decoder1/fcn1/LeakyRelu╝
%decoder2/fcn1/MatMul_1/ReadVariableOpReadVariableOp,decoder2_fcn1_matmul_readvariableop_resource*
_output_shapes
:	`А	*
dtype02'
%decoder2/fcn1/MatMul_1/ReadVariableOp╚
decoder2/fcn1/MatMul_1MatMul*encoder2/fcn2/mean/LeakyRelu:activations:0-decoder2/fcn1/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:         А	2
decoder2/fcn1/MatMul_1╗
&decoder2/fcn1/BiasAdd_1/ReadVariableOpReadVariableOp-decoder2_fcn1_biasadd_readvariableop_resource*
_output_shapes	
:А	*
dtype02(
&decoder2/fcn1/BiasAdd_1/ReadVariableOp┬
decoder2/fcn1/BiasAdd_1BiasAdd decoder2/fcn1/MatMul_1:product:0.decoder2/fcn1/BiasAdd_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:         А	2
decoder2/fcn1/BiasAdd_1П
decoder2/fcn1/LeakyRelu_1	LeakyRelu decoder2/fcn1/BiasAdd_1:output:0*(
_output_shapes
:         А	2
decoder2/fcn1/LeakyRelu_1╝
%decoder1/fcn1/MatMul_1/ReadVariableOpReadVariableOp,decoder1_fcn1_matmul_readvariableop_resource*
_output_shapes
:	dА	*
dtype02'
%decoder1/fcn1/MatMul_1/ReadVariableOp╝
decoder1/fcn1/MatMul_1MatMultf.__operators__.add/AddV2:z:0-decoder1/fcn1/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:         А	2
decoder1/fcn1/MatMul_1╗
&decoder1/fcn1/BiasAdd_1/ReadVariableOpReadVariableOp-decoder1_fcn1_biasadd_readvariableop_resource*
_output_shapes	
:А	*
dtype02(
&decoder1/fcn1/BiasAdd_1/ReadVariableOp┬
decoder1/fcn1/BiasAdd_1BiasAdd decoder1/fcn1/MatMul_1:product:0.decoder1/fcn1/BiasAdd_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:         А	2
decoder1/fcn1/BiasAdd_1П
decoder1/fcn1/LeakyRelu_1	LeakyRelu decoder1/fcn1/BiasAdd_1:output:0*(
_output_shapes
:         А	2
decoder1/fcn1/LeakyRelu_1Ч
tf.nn.softmax_1/SoftmaxSoftmaxtf.math.truediv_1/truediv:z:0*
T0*0
_output_shapes
:                  2
tf.nn.softmax_1/SoftmaxС
tf.nn.softmax/SoftmaxSoftmaxtf.math.truediv/truediv:z:0*
T0*0
_output_shapes
:                  2
tf.nn.softmax/Softmaxy
tf.math.multiply_9/Mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *  HB2
tf.math.multiply_9/Mul/yЧ
tf.math.multiply_9/MulMulinputs_4!tf.math.multiply_9/Mul/y:output:0*
T0*(
_output_shapes
:         А2
tf.math.multiply_9/Mul╣
#decoder2/fcn2/MatMul/ReadVariableOpReadVariableOp,decoder2_fcn2_matmul_readvariableop_resource* 
_output_shapes
:
А	А*
dtype02%
#decoder2/fcn2/MatMul/ReadVariableOp╜
decoder2/fcn2/MatMulMatMul%decoder2/fcn1/LeakyRelu:activations:0+decoder2/fcn2/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         А2
decoder2/fcn2/MatMul╖
$decoder2/fcn2/BiasAdd/ReadVariableOpReadVariableOp-decoder2_fcn2_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype02&
$decoder2/fcn2/BiasAdd/ReadVariableOp║
decoder2/fcn2/BiasAddBiasAdddecoder2/fcn2/MatMul:product:0,decoder2/fcn2/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         А2
decoder2/fcn2/BiasAddy
tf.math.multiply_8/Mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *  HB2
tf.math.multiply_8/Mul/yЧ
tf.math.multiply_8/MulMulinputs_2!tf.math.multiply_8/Mul/y:output:0*
T0*(
_output_shapes
:         А2
tf.math.multiply_8/Mul╣
#decoder1/fcn2/MatMul/ReadVariableOpReadVariableOp,decoder1_fcn2_matmul_readvariableop_resource* 
_output_shapes
:
А	А*
dtype02%
#decoder1/fcn2/MatMul/ReadVariableOp╜
decoder1/fcn2/MatMulMatMul%decoder1/fcn1/LeakyRelu:activations:0+decoder1/fcn2/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         А2
decoder1/fcn2/MatMul╖
$decoder1/fcn2/BiasAdd/ReadVariableOpReadVariableOp-decoder1_fcn2_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype02&
$decoder1/fcn2/BiasAdd/ReadVariableOp║
decoder1/fcn2/BiasAddBiasAdddecoder1/fcn2/MatMul:product:0,decoder1/fcn2/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         А2
decoder1/fcn2/BiasAdd╜
%decoder2/fcn2/MatMul_1/ReadVariableOpReadVariableOp,decoder2_fcn2_matmul_readvariableop_resource* 
_output_shapes
:
А	А*
dtype02'
%decoder2/fcn2/MatMul_1/ReadVariableOp┼
decoder2/fcn2/MatMul_1MatMul'decoder2/fcn1/LeakyRelu_1:activations:0-decoder2/fcn2/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:         А2
decoder2/fcn2/MatMul_1╗
&decoder2/fcn2/BiasAdd_1/ReadVariableOpReadVariableOp-decoder2_fcn2_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype02(
&decoder2/fcn2/BiasAdd_1/ReadVariableOp┬
decoder2/fcn2/BiasAdd_1BiasAdd decoder2/fcn2/MatMul_1:product:0.decoder2/fcn2/BiasAdd_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:         А2
decoder2/fcn2/BiasAdd_1╜
%decoder1/fcn2/MatMul_1/ReadVariableOpReadVariableOp,decoder1_fcn2_matmul_readvariableop_resource* 
_output_shapes
:
А	А*
dtype02'
%decoder1/fcn2/MatMul_1/ReadVariableOp┼
decoder1/fcn2/MatMul_1MatMul'decoder1/fcn1/LeakyRelu_1:activations:0-decoder1/fcn2/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:         А2
decoder1/fcn2/MatMul_1╗
&decoder1/fcn2/BiasAdd_1/ReadVariableOpReadVariableOp-decoder1_fcn2_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype02(
&decoder1/fcn2/BiasAdd_1/ReadVariableOp┬
decoder1/fcn2/BiasAdd_1BiasAdd decoder1/fcn2/MatMul_1:product:0.decoder1/fcn2/BiasAdd_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:         А2
decoder1/fcn2/BiasAdd_1└
tf.math.truediv_3/truedivRealDiv!tf.nn.softmax_1/Softmax:softmax:0tf.nn.softmax/Softmax:softmax:0*
T0*0
_output_shapes
:                  2
tf.math.truediv_3/truediv└
tf.math.truediv_2/truedivRealDivtf.nn.softmax/Softmax:softmax:0!tf.nn.softmax_1/Softmax:softmax:0*
T0*0
_output_shapes
:                  2
tf.math.truediv_2/truedivж
tf.math.subtract_5/SubSubtf.math.multiply_9/Mul:z:0decoder2/fcn2/BiasAdd:output:0*
T0*(
_output_shapes
:         А2
tf.math.subtract_5/Subж
tf.math.subtract_4/SubSubtf.math.multiply_8/Mul:z:0decoder1/fcn2/BiasAdd:output:0*
T0*(
_output_shapes
:         А2
tf.math.subtract_4/Suby
tf.math.multiply_1/Mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *  HB2
tf.math.multiply_1/Mul/yЧ
tf.math.multiply_1/MulMulinputs_1!tf.math.multiply_1/Mul/y:output:0*
T0*(
_output_shapes
:         А2
tf.math.multiply_1/Mulu
tf.math.multiply/Mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *  HB2
tf.math.multiply/Mul/yС
tf.math.multiply/MulMulinputs_0tf.math.multiply/Mul/y:output:0*
T0*(
_output_shapes
:         А2
tf.math.multiply/MulЗ
tf.math.log_1/LogLogtf.math.truediv_3/truediv:z:0*
T0*0
_output_shapes
:                  2
tf.math.log_1/LogГ
tf.math.log/LogLogtf.math.truediv_2/truediv:z:0*
T0*0
_output_shapes
:                  2
tf.math.log/LogЛ
tf.math.square_9/SquareSquaretf.math.subtract_5/Sub:z:0*
T0*(
_output_shapes
:         А2
tf.math.square_9/SquareЛ
tf.math.square_8/SquareSquaretf.math.subtract_4/Sub:z:0*
T0*(
_output_shapes
:         А2
tf.math.square_8/Squareи
tf.math.subtract_2/SubSubtf.math.multiply_1/Mul:z:0 decoder2/fcn2/BiasAdd_1:output:0*
T0*(
_output_shapes
:         А2
tf.math.subtract_2/Subж
tf.math.subtract_1/SubSubtf.math.multiply/Mul:z:0 decoder1/fcn2/BiasAdd_1:output:0*
T0*(
_output_shapes
:         А2
tf.math.subtract_1/Subм
tf.math.multiply_5/MulMul!tf.nn.softmax_1/Softmax:softmax:0tf.math.log_1/Log:y:0*
T0*0
_output_shapes
:                  2
tf.math.multiply_5/Mulи
tf.math.multiply_4/MulMultf.nn.softmax/Softmax:softmax:0tf.math.log/Log:y:0*
T0*0
_output_shapes
:                  2
tf.math.multiply_4/MulЛ
tf.math.reduce_mean_3/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2
tf.math.reduce_mean_3/Constд
tf.math.reduce_mean_3/MeanMeantf.math.square_8/Square:y:0$tf.math.reduce_mean_3/Const:output:0*
T0*
_output_shapes
: 2
tf.math.reduce_mean_3/MeanЛ
tf.math.reduce_mean_4/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2
tf.math.reduce_mean_4/Constд
tf.math.reduce_mean_4/MeanMeantf.math.square_9/Square:y:0$tf.math.reduce_mean_4/Const:output:0*
T0*
_output_shapes
: 2
tf.math.reduce_mean_4/MeanЛ
tf.math.square_1/SquareSquaretf.math.subtract_2/Sub:z:0*
T0*(
_output_shapes
:         А2
tf.math.square_1/SquareЗ
tf.math.square/SquareSquaretf.math.subtract_1/Sub:z:0*
T0*(
_output_shapes
:         А2
tf.math.square/Square╖
#encoder2/fcn3/MatMul/ReadVariableOpReadVariableOp,encoder2_fcn3_matmul_readvariableop_resource*
_output_shapes

:`P*
dtype02%
#encoder2/fcn3/MatMul/ReadVariableOp┴
encoder2/fcn3/MatMulMatMul*encoder2/fcn2/mean/LeakyRelu:activations:0+encoder2/fcn3/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         P2
encoder2/fcn3/MatMul╢
$encoder2/fcn3/BiasAdd/ReadVariableOpReadVariableOp-encoder2_fcn3_biasadd_readvariableop_resource*
_output_shapes
:P*
dtype02&
$encoder2/fcn3/BiasAdd/ReadVariableOp╣
encoder2/fcn3/BiasAddBiasAddencoder2/fcn3/MatMul:product:0,encoder2/fcn3/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         P2
encoder2/fcn3/BiasAdd╖
#encoder1/fcn3/MatMul/ReadVariableOpReadVariableOp,encoder1_fcn3_matmul_readvariableop_resource*
_output_shapes

:dP*
dtype02%
#encoder1/fcn3/MatMul/ReadVariableOp├
encoder1/fcn3/MatMulMatMul,encoder1/fcn2/mean/LeakyRelu_1:activations:0+encoder1/fcn3/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         P2
encoder1/fcn3/MatMul╢
$encoder1/fcn3/BiasAdd/ReadVariableOpReadVariableOp-encoder1_fcn3_biasadd_readvariableop_resource*
_output_shapes
:P*
dtype02&
$encoder1/fcn3/BiasAdd/ReadVariableOp╣
encoder1/fcn3/BiasAddBiasAddencoder1/fcn3/MatMul:product:0,encoder1/fcn3/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         P2
encoder1/fcn3/BiasAddг
*tf.math.reduce_sum_4/Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
valueB :
         2,
*tf.math.reduce_sum_4/Sum/reduction_indices║
tf.math.reduce_sum_4/SumSumtf.math.multiply_4/Mul:z:03tf.math.reduce_sum_4/Sum/reduction_indices:output:0*
T0*#
_output_shapes
:         2
tf.math.reduce_sum_4/Sumг
*tf.math.reduce_sum_5/Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
valueB :
         2,
*tf.math.reduce_sum_5/Sum/reduction_indices║
tf.math.reduce_sum_5/SumSumtf.math.multiply_5/Mul:z:03tf.math.reduce_sum_5/Sum/reduction_indices:output:0*
T0*#
_output_shapes
:         2
tf.math.reduce_sum_5/Sum▓
tf.__operators__.add_16/AddV2AddV2#tf.math.reduce_mean_3/Mean:output:0#tf.math.reduce_mean_4/Mean:output:0*
T0*
_output_shapes
: 2
tf.__operators__.add_16/AddV2З
tf.math.reduce_mean/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2
tf.math.reduce_mean/ConstЬ
tf.math.reduce_mean/MeanMeantf.math.square/Square:y:0"tf.math.reduce_mean/Const:output:0*
T0*
_output_shapes
: 2
tf.math.reduce_mean/MeanЛ
tf.math.reduce_mean_1/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2
tf.math.reduce_mean_1/Constд
tf.math.reduce_mean_1/MeanMeantf.math.square_1/Square:y:0$tf.math.reduce_mean_1/Const:output:0*
T0*
_output_shapes
: 2
tf.math.reduce_mean_1/MeanО
tf.math.square_7/SquareSquareencoder2/fcn3/BiasAdd:output:0*
T0*'
_output_shapes
:         P2
tf.math.square_7/SquareО
tf.math.square_6/SquareSquareencoder1/fcn3/BiasAdd:output:0*
T0*'
_output_shapes
:         P2
tf.math.square_6/Square╣
tf.__operators__.add_1/AddV2AddV2!tf.math.reduce_sum_4/Sum:output:0!tf.math.reduce_sum_5/Sum:output:0*
T0*#
_output_shapes
:         2
tf.__operators__.add_1/AddV2Ж
tf.math.multiply_22/MulMulunknown!tf.__operators__.add_16/AddV2:z:0*
T0*
_output_shapes
: 2
tf.math.multiply_22/Mul░
tf.__operators__.add_15/AddV2AddV2!tf.math.reduce_mean/Mean:output:0#tf.math.reduce_mean_1/Mean:output:0*
T0*
_output_shapes
: 2
tf.__operators__.add_15/AddV2г
*tf.math.reduce_sum_8/Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
valueB :
         2,
*tf.math.reduce_sum_8/Sum/reduction_indices╗
tf.math.reduce_sum_8/SumSumtf.math.square_7/Square:y:03tf.math.reduce_sum_8/Sum/reduction_indices:output:0*
T0*#
_output_shapes
:         2
tf.math.reduce_sum_8/Sumг
*tf.math.reduce_sum_7/Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
valueB :
         2,
*tf.math.reduce_sum_7/Sum/reduction_indices╗
tf.math.reduce_sum_7/SumSumtf.math.square_6/Square:y:03tf.math.reduce_sum_7/Sum/reduction_indices:output:0*
T0*#
_output_shapes
:         2
tf.math.reduce_sum_7/SumД
tf.math.reduce_mean_2/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2
tf.math.reduce_mean_2/Constй
tf.math.reduce_mean_2/MeanMean tf.__operators__.add_1/AddV2:z:0$tf.math.reduce_mean_2/Const:output:0*
T0*
_output_shapes
: 2
tf.math.reduce_mean_2/MeanБ
tf.math.truediv_21/truediv/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
tf.math.truediv_21/truediv/yо
tf.math.truediv_21/truedivRealDiv!tf.__operators__.add_15/AddV2:z:0%tf.math.truediv_21/truediv/y:output:0*
T0*
_output_shapes
: 2
tf.math.truediv_21/truedivБ
tf.math.truediv_22/truediv/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
tf.math.truediv_22/truediv/yи
tf.math.truediv_22/truedivRealDivtf.math.multiply_22/Mul:z:0%tf.math.truediv_22/truediv/y:output:0*
T0*
_output_shapes
: 2
tf.math.truediv_22/truedivГ
tf.math.sqrt_4/SqrtSqrt!tf.math.reduce_sum_7/Sum:output:0*
T0*#
_output_shapes
:         2
tf.math.sqrt_4/SqrtГ
tf.math.sqrt_5/SqrtSqrt!tf.math.reduce_sum_8/Sum:output:0*
T0*#
_output_shapes
:         2
tf.math.sqrt_5/Sqrtй
tf.math.multiply_6/MulMulencoder1/fcn3/BiasAdd:output:0encoder2/fcn3/BiasAdd:output:0*
T0*'
_output_shapes
:         P2
tf.math.multiply_6/Mulи
tf.__operators__.add_17/AddV2AddV2tf.math.truediv_21/truediv:z:0tf.math.truediv_22/truediv:z:0*
T0*
_output_shapes
: 2
tf.__operators__.add_17/AddV2{
tf.math.multiply_23/Mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2
tf.math.multiply_23/Mul/yг
tf.math.multiply_23/MulMul#tf.math.reduce_mean_2/Mean:output:0"tf.math.multiply_23/Mul/y:output:0*
T0*
_output_shapes
: 2
tf.math.multiply_23/Mulг
*tf.math.reduce_sum_6/Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
valueB :
         2,
*tf.math.reduce_sum_6/Sum/reduction_indices║
tf.math.reduce_sum_6/SumSumtf.math.multiply_6/Mul:z:03tf.math.reduce_sum_6/Sum/reduction_indices:output:0*
T0*#
_output_shapes
:         2
tf.math.reduce_sum_6/SumЧ
tf.math.multiply_7/MulMultf.math.sqrt_4/Sqrt:y:0tf.math.sqrt_5/Sqrt:y:0*
T0*#
_output_shapes
:         2
tf.math.multiply_7/Mulи
tf.__operators__.add_18/AddV2AddV2!tf.__operators__.add_17/AddV2:z:0tf.math.multiply_23/Mul:z:0*
T0*
_output_shapes
: 2
tf.__operators__.add_18/AddV2о
tf.math.truediv_4/truedivRealDiv!tf.math.reduce_sum_6/Sum:output:0tf.math.multiply_7/Mul:z:0*
T0*#
_output_shapes
:         2
tf.math.truediv_4/truedivЗ
IdentityIdentitytf.math.truediv_4/truediv:z:0%^decoder1/fcn1/BiasAdd/ReadVariableOp'^decoder1/fcn1/BiasAdd_1/ReadVariableOp$^decoder1/fcn1/MatMul/ReadVariableOp&^decoder1/fcn1/MatMul_1/ReadVariableOp%^decoder1/fcn2/BiasAdd/ReadVariableOp'^decoder1/fcn2/BiasAdd_1/ReadVariableOp$^decoder1/fcn2/MatMul/ReadVariableOp&^decoder1/fcn2/MatMul_1/ReadVariableOp+^decoder1/layer_gene/BiasAdd/ReadVariableOp-^decoder1/layer_gene/BiasAdd_1/ReadVariableOp*^decoder1/layer_gene/MatMul/ReadVariableOp,^decoder1/layer_gene/MatMul_1/ReadVariableOp%^decoder2/fcn1/BiasAdd/ReadVariableOp'^decoder2/fcn1/BiasAdd_1/ReadVariableOp$^decoder2/fcn1/MatMul/ReadVariableOp&^decoder2/fcn1/MatMul_1/ReadVariableOp%^decoder2/fcn2/BiasAdd/ReadVariableOp'^decoder2/fcn2/BiasAdd_1/ReadVariableOp$^decoder2/fcn2/MatMul/ReadVariableOp&^decoder2/fcn2/MatMul_1/ReadVariableOp%^encoder1/fcn1/BiasAdd/ReadVariableOp'^encoder1/fcn1/BiasAdd_1/ReadVariableOp$^encoder1/fcn1/MatMul/ReadVariableOp&^encoder1/fcn1/MatMul_1/ReadVariableOp*^encoder1/fcn2/mean/BiasAdd/ReadVariableOp,^encoder1/fcn2/mean/BiasAdd_1/ReadVariableOp)^encoder1/fcn2/mean/MatMul/ReadVariableOp+^encoder1/fcn2/mean/MatMul_1/ReadVariableOp%^encoder1/fcn3/BiasAdd/ReadVariableOp$^encoder1/fcn3/MatMul/ReadVariableOp*^encoder1/fcn3_gene/BiasAdd/ReadVariableOp,^encoder1/fcn3_gene/BiasAdd_1/ReadVariableOp)^encoder1/fcn3_gene/MatMul/ReadVariableOp+^encoder1/fcn3_gene/MatMul_1/ReadVariableOp%^encoder2/fcn1/BiasAdd/ReadVariableOp'^encoder2/fcn1/BiasAdd_1/ReadVariableOp$^encoder2/fcn1/MatMul/ReadVariableOp&^encoder2/fcn1/MatMul_1/ReadVariableOp*^encoder2/fcn2/mean/BiasAdd/ReadVariableOp,^encoder2/fcn2/mean/BiasAdd_1/ReadVariableOp)^encoder2/fcn2/mean/MatMul/ReadVariableOp+^encoder2/fcn2/mean/MatMul_1/ReadVariableOp%^encoder2/fcn3/BiasAdd/ReadVariableOp$^encoder2/fcn3/MatMul/ReadVariableOp*
T0*#
_output_shapes
:         2

IdentityВ

Identity_1Identity!tf.__operators__.add_18/AddV2:z:0%^decoder1/fcn1/BiasAdd/ReadVariableOp'^decoder1/fcn1/BiasAdd_1/ReadVariableOp$^decoder1/fcn1/MatMul/ReadVariableOp&^decoder1/fcn1/MatMul_1/ReadVariableOp%^decoder1/fcn2/BiasAdd/ReadVariableOp'^decoder1/fcn2/BiasAdd_1/ReadVariableOp$^decoder1/fcn2/MatMul/ReadVariableOp&^decoder1/fcn2/MatMul_1/ReadVariableOp+^decoder1/layer_gene/BiasAdd/ReadVariableOp-^decoder1/layer_gene/BiasAdd_1/ReadVariableOp*^decoder1/layer_gene/MatMul/ReadVariableOp,^decoder1/layer_gene/MatMul_1/ReadVariableOp%^decoder2/fcn1/BiasAdd/ReadVariableOp'^decoder2/fcn1/BiasAdd_1/ReadVariableOp$^decoder2/fcn1/MatMul/ReadVariableOp&^decoder2/fcn1/MatMul_1/ReadVariableOp%^decoder2/fcn2/BiasAdd/ReadVariableOp'^decoder2/fcn2/BiasAdd_1/ReadVariableOp$^decoder2/fcn2/MatMul/ReadVariableOp&^decoder2/fcn2/MatMul_1/ReadVariableOp%^encoder1/fcn1/BiasAdd/ReadVariableOp'^encoder1/fcn1/BiasAdd_1/ReadVariableOp$^encoder1/fcn1/MatMul/ReadVariableOp&^encoder1/fcn1/MatMul_1/ReadVariableOp*^encoder1/fcn2/mean/BiasAdd/ReadVariableOp,^encoder1/fcn2/mean/BiasAdd_1/ReadVariableOp)^encoder1/fcn2/mean/MatMul/ReadVariableOp+^encoder1/fcn2/mean/MatMul_1/ReadVariableOp%^encoder1/fcn3/BiasAdd/ReadVariableOp$^encoder1/fcn3/MatMul/ReadVariableOp*^encoder1/fcn3_gene/BiasAdd/ReadVariableOp,^encoder1/fcn3_gene/BiasAdd_1/ReadVariableOp)^encoder1/fcn3_gene/MatMul/ReadVariableOp+^encoder1/fcn3_gene/MatMul_1/ReadVariableOp%^encoder2/fcn1/BiasAdd/ReadVariableOp'^encoder2/fcn1/BiasAdd_1/ReadVariableOp$^encoder2/fcn1/MatMul/ReadVariableOp&^encoder2/fcn1/MatMul_1/ReadVariableOp*^encoder2/fcn2/mean/BiasAdd/ReadVariableOp,^encoder2/fcn2/mean/BiasAdd_1/ReadVariableOp)^encoder2/fcn2/mean/MatMul/ReadVariableOp+^encoder2/fcn2/mean/MatMul_1/ReadVariableOp%^encoder2/fcn3/BiasAdd/ReadVariableOp$^encoder2/fcn3/MatMul/ReadVariableOp*
T0*
_output_shapes
: 2

Identity_1Р

Identity_2Identityencoder1/fcn3/BiasAdd:output:0%^decoder1/fcn1/BiasAdd/ReadVariableOp'^decoder1/fcn1/BiasAdd_1/ReadVariableOp$^decoder1/fcn1/MatMul/ReadVariableOp&^decoder1/fcn1/MatMul_1/ReadVariableOp%^decoder1/fcn2/BiasAdd/ReadVariableOp'^decoder1/fcn2/BiasAdd_1/ReadVariableOp$^decoder1/fcn2/MatMul/ReadVariableOp&^decoder1/fcn2/MatMul_1/ReadVariableOp+^decoder1/layer_gene/BiasAdd/ReadVariableOp-^decoder1/layer_gene/BiasAdd_1/ReadVariableOp*^decoder1/layer_gene/MatMul/ReadVariableOp,^decoder1/layer_gene/MatMul_1/ReadVariableOp%^decoder2/fcn1/BiasAdd/ReadVariableOp'^decoder2/fcn1/BiasAdd_1/ReadVariableOp$^decoder2/fcn1/MatMul/ReadVariableOp&^decoder2/fcn1/MatMul_1/ReadVariableOp%^decoder2/fcn2/BiasAdd/ReadVariableOp'^decoder2/fcn2/BiasAdd_1/ReadVariableOp$^decoder2/fcn2/MatMul/ReadVariableOp&^decoder2/fcn2/MatMul_1/ReadVariableOp%^encoder1/fcn1/BiasAdd/ReadVariableOp'^encoder1/fcn1/BiasAdd_1/ReadVariableOp$^encoder1/fcn1/MatMul/ReadVariableOp&^encoder1/fcn1/MatMul_1/ReadVariableOp*^encoder1/fcn2/mean/BiasAdd/ReadVariableOp,^encoder1/fcn2/mean/BiasAdd_1/ReadVariableOp)^encoder1/fcn2/mean/MatMul/ReadVariableOp+^encoder1/fcn2/mean/MatMul_1/ReadVariableOp%^encoder1/fcn3/BiasAdd/ReadVariableOp$^encoder1/fcn3/MatMul/ReadVariableOp*^encoder1/fcn3_gene/BiasAdd/ReadVariableOp,^encoder1/fcn3_gene/BiasAdd_1/ReadVariableOp)^encoder1/fcn3_gene/MatMul/ReadVariableOp+^encoder1/fcn3_gene/MatMul_1/ReadVariableOp%^encoder2/fcn1/BiasAdd/ReadVariableOp'^encoder2/fcn1/BiasAdd_1/ReadVariableOp$^encoder2/fcn1/MatMul/ReadVariableOp&^encoder2/fcn1/MatMul_1/ReadVariableOp*^encoder2/fcn2/mean/BiasAdd/ReadVariableOp,^encoder2/fcn2/mean/BiasAdd_1/ReadVariableOp)^encoder2/fcn2/mean/MatMul/ReadVariableOp+^encoder2/fcn2/mean/MatMul_1/ReadVariableOp%^encoder2/fcn3/BiasAdd/ReadVariableOp$^encoder2/fcn3/MatMul/ReadVariableOp*
T0*'
_output_shapes
:         P2

Identity_2Р

Identity_3Identityencoder2/fcn3/BiasAdd:output:0%^decoder1/fcn1/BiasAdd/ReadVariableOp'^decoder1/fcn1/BiasAdd_1/ReadVariableOp$^decoder1/fcn1/MatMul/ReadVariableOp&^decoder1/fcn1/MatMul_1/ReadVariableOp%^decoder1/fcn2/BiasAdd/ReadVariableOp'^decoder1/fcn2/BiasAdd_1/ReadVariableOp$^decoder1/fcn2/MatMul/ReadVariableOp&^decoder1/fcn2/MatMul_1/ReadVariableOp+^decoder1/layer_gene/BiasAdd/ReadVariableOp-^decoder1/layer_gene/BiasAdd_1/ReadVariableOp*^decoder1/layer_gene/MatMul/ReadVariableOp,^decoder1/layer_gene/MatMul_1/ReadVariableOp%^decoder2/fcn1/BiasAdd/ReadVariableOp'^decoder2/fcn1/BiasAdd_1/ReadVariableOp$^decoder2/fcn1/MatMul/ReadVariableOp&^decoder2/fcn1/MatMul_1/ReadVariableOp%^decoder2/fcn2/BiasAdd/ReadVariableOp'^decoder2/fcn2/BiasAdd_1/ReadVariableOp$^decoder2/fcn2/MatMul/ReadVariableOp&^decoder2/fcn2/MatMul_1/ReadVariableOp%^encoder1/fcn1/BiasAdd/ReadVariableOp'^encoder1/fcn1/BiasAdd_1/ReadVariableOp$^encoder1/fcn1/MatMul/ReadVariableOp&^encoder1/fcn1/MatMul_1/ReadVariableOp*^encoder1/fcn2/mean/BiasAdd/ReadVariableOp,^encoder1/fcn2/mean/BiasAdd_1/ReadVariableOp)^encoder1/fcn2/mean/MatMul/ReadVariableOp+^encoder1/fcn2/mean/MatMul_1/ReadVariableOp%^encoder1/fcn3/BiasAdd/ReadVariableOp$^encoder1/fcn3/MatMul/ReadVariableOp*^encoder1/fcn3_gene/BiasAdd/ReadVariableOp,^encoder1/fcn3_gene/BiasAdd_1/ReadVariableOp)^encoder1/fcn3_gene/MatMul/ReadVariableOp+^encoder1/fcn3_gene/MatMul_1/ReadVariableOp%^encoder2/fcn1/BiasAdd/ReadVariableOp'^encoder2/fcn1/BiasAdd_1/ReadVariableOp$^encoder2/fcn1/MatMul/ReadVariableOp&^encoder2/fcn1/MatMul_1/ReadVariableOp*^encoder2/fcn2/mean/BiasAdd/ReadVariableOp,^encoder2/fcn2/mean/BiasAdd_1/ReadVariableOp)^encoder2/fcn2/mean/MatMul/ReadVariableOp+^encoder2/fcn2/mean/MatMul_1/ReadVariableOp%^encoder2/fcn3/BiasAdd/ReadVariableOp$^encoder2/fcn3/MatMul/ReadVariableOp*
T0*'
_output_shapes
:         P2

Identity_3"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0*(
_construction_contextkEagerRuntime*┐
_input_shapesн
к:         А:         А:         А:         А:         А:         А: : : : : : : : : : : : : : : : : : : : : : : : : 2L
$decoder1/fcn1/BiasAdd/ReadVariableOp$decoder1/fcn1/BiasAdd/ReadVariableOp2P
&decoder1/fcn1/BiasAdd_1/ReadVariableOp&decoder1/fcn1/BiasAdd_1/ReadVariableOp2J
#decoder1/fcn1/MatMul/ReadVariableOp#decoder1/fcn1/MatMul/ReadVariableOp2N
%decoder1/fcn1/MatMul_1/ReadVariableOp%decoder1/fcn1/MatMul_1/ReadVariableOp2L
$decoder1/fcn2/BiasAdd/ReadVariableOp$decoder1/fcn2/BiasAdd/ReadVariableOp2P
&decoder1/fcn2/BiasAdd_1/ReadVariableOp&decoder1/fcn2/BiasAdd_1/ReadVariableOp2J
#decoder1/fcn2/MatMul/ReadVariableOp#decoder1/fcn2/MatMul/ReadVariableOp2N
%decoder1/fcn2/MatMul_1/ReadVariableOp%decoder1/fcn2/MatMul_1/ReadVariableOp2X
*decoder1/layer_gene/BiasAdd/ReadVariableOp*decoder1/layer_gene/BiasAdd/ReadVariableOp2\
,decoder1/layer_gene/BiasAdd_1/ReadVariableOp,decoder1/layer_gene/BiasAdd_1/ReadVariableOp2V
)decoder1/layer_gene/MatMul/ReadVariableOp)decoder1/layer_gene/MatMul/ReadVariableOp2Z
+decoder1/layer_gene/MatMul_1/ReadVariableOp+decoder1/layer_gene/MatMul_1/ReadVariableOp2L
$decoder2/fcn1/BiasAdd/ReadVariableOp$decoder2/fcn1/BiasAdd/ReadVariableOp2P
&decoder2/fcn1/BiasAdd_1/ReadVariableOp&decoder2/fcn1/BiasAdd_1/ReadVariableOp2J
#decoder2/fcn1/MatMul/ReadVariableOp#decoder2/fcn1/MatMul/ReadVariableOp2N
%decoder2/fcn1/MatMul_1/ReadVariableOp%decoder2/fcn1/MatMul_1/ReadVariableOp2L
$decoder2/fcn2/BiasAdd/ReadVariableOp$decoder2/fcn2/BiasAdd/ReadVariableOp2P
&decoder2/fcn2/BiasAdd_1/ReadVariableOp&decoder2/fcn2/BiasAdd_1/ReadVariableOp2J
#decoder2/fcn2/MatMul/ReadVariableOp#decoder2/fcn2/MatMul/ReadVariableOp2N
%decoder2/fcn2/MatMul_1/ReadVariableOp%decoder2/fcn2/MatMul_1/ReadVariableOp2L
$encoder1/fcn1/BiasAdd/ReadVariableOp$encoder1/fcn1/BiasAdd/ReadVariableOp2P
&encoder1/fcn1/BiasAdd_1/ReadVariableOp&encoder1/fcn1/BiasAdd_1/ReadVariableOp2J
#encoder1/fcn1/MatMul/ReadVariableOp#encoder1/fcn1/MatMul/ReadVariableOp2N
%encoder1/fcn1/MatMul_1/ReadVariableOp%encoder1/fcn1/MatMul_1/ReadVariableOp2V
)encoder1/fcn2/mean/BiasAdd/ReadVariableOp)encoder1/fcn2/mean/BiasAdd/ReadVariableOp2Z
+encoder1/fcn2/mean/BiasAdd_1/ReadVariableOp+encoder1/fcn2/mean/BiasAdd_1/ReadVariableOp2T
(encoder1/fcn2/mean/MatMul/ReadVariableOp(encoder1/fcn2/mean/MatMul/ReadVariableOp2X
*encoder1/fcn2/mean/MatMul_1/ReadVariableOp*encoder1/fcn2/mean/MatMul_1/ReadVariableOp2L
$encoder1/fcn3/BiasAdd/ReadVariableOp$encoder1/fcn3/BiasAdd/ReadVariableOp2J
#encoder1/fcn3/MatMul/ReadVariableOp#encoder1/fcn3/MatMul/ReadVariableOp2V
)encoder1/fcn3_gene/BiasAdd/ReadVariableOp)encoder1/fcn3_gene/BiasAdd/ReadVariableOp2Z
+encoder1/fcn3_gene/BiasAdd_1/ReadVariableOp+encoder1/fcn3_gene/BiasAdd_1/ReadVariableOp2T
(encoder1/fcn3_gene/MatMul/ReadVariableOp(encoder1/fcn3_gene/MatMul/ReadVariableOp2X
*encoder1/fcn3_gene/MatMul_1/ReadVariableOp*encoder1/fcn3_gene/MatMul_1/ReadVariableOp2L
$encoder2/fcn1/BiasAdd/ReadVariableOp$encoder2/fcn1/BiasAdd/ReadVariableOp2P
&encoder2/fcn1/BiasAdd_1/ReadVariableOp&encoder2/fcn1/BiasAdd_1/ReadVariableOp2J
#encoder2/fcn1/MatMul/ReadVariableOp#encoder2/fcn1/MatMul/ReadVariableOp2N
%encoder2/fcn1/MatMul_1/ReadVariableOp%encoder2/fcn1/MatMul_1/ReadVariableOp2V
)encoder2/fcn2/mean/BiasAdd/ReadVariableOp)encoder2/fcn2/mean/BiasAdd/ReadVariableOp2Z
+encoder2/fcn2/mean/BiasAdd_1/ReadVariableOp+encoder2/fcn2/mean/BiasAdd_1/ReadVariableOp2T
(encoder2/fcn2/mean/MatMul/ReadVariableOp(encoder2/fcn2/mean/MatMul/ReadVariableOp2X
*encoder2/fcn2/mean/MatMul_1/ReadVariableOp*encoder2/fcn2/mean/MatMul_1/ReadVariableOp2L
$encoder2/fcn3/BiasAdd/ReadVariableOp$encoder2/fcn3/BiasAdd/ReadVariableOp2J
#encoder2/fcn3/MatMul/ReadVariableOp#encoder2/fcn3/MatMul/ReadVariableOp:R N
(
_output_shapes
:         А
"
_user_specified_name
inputs/0:RN
(
_output_shapes
:         А
"
_user_specified_name
inputs/1:RN
(
_output_shapes
:         А
"
_user_specified_name
inputs/2:RN
(
_output_shapes
:         А
"
_user_specified_name
inputs/3:RN
(
_output_shapes
:         А
"
_user_specified_name
inputs/4:RN
(
_output_shapes
:         А
"
_user_specified_name
inputs/5:

_output_shapes
: 
нН
Ж
C__inference_model_layer_call_and_return_conditional_losses_40050737
input_1
input_2
input_3
input_4
input_5
input_6)
encoder2_fcn1_40050552:	Аx$
encoder2_fcn1_40050554:x-
encoder2_fcn2_mean_40050557:x`)
encoder2_fcn2_mean_40050559:`.
encoder1_fcn3_gene_40050562:	Аd)
encoder1_fcn3_gene_40050564:d)
encoder1_fcn1_40050567:	Аd$
encoder1_fcn1_40050569:d-
encoder1_fcn2_mean_40050592:dd)
encoder1_fcn2_mean_40050594:d.
decoder1_layer_gene_40050607:dd*
decoder1_layer_gene_40050609:d)
decoder2_fcn1_40050626:	`А	%
decoder2_fcn1_40050628:	А	)
decoder1_fcn1_40050631:	dА	%
decoder1_fcn1_40050633:	А	*
decoder2_fcn2_40050646:
А	А%
decoder2_fcn2_40050648:	А*
decoder1_fcn2_40050653:
А	А%
decoder1_fcn2_40050655:	А(
encoder2_fcn3_40050686:`P$
encoder2_fcn3_40050688:P(
encoder1_fcn3_40050691:dP$
encoder1_fcn3_40050693:P
unknown
identity

identity_1

identity_2

identity_3Ив%decoder1/fcn1/StatefulPartitionedCallв'decoder1/fcn1/StatefulPartitionedCall_1в%decoder1/fcn2/StatefulPartitionedCallв'decoder1/fcn2/StatefulPartitionedCall_1в+decoder1/layer_gene/StatefulPartitionedCallв-decoder1/layer_gene/StatefulPartitionedCall_1в%decoder2/fcn1/StatefulPartitionedCallв'decoder2/fcn1/StatefulPartitionedCall_1в%decoder2/fcn2/StatefulPartitionedCallв'decoder2/fcn2/StatefulPartitionedCall_1в%encoder1/fcn1/StatefulPartitionedCallв'encoder1/fcn1/StatefulPartitionedCall_1в*encoder1/fcn2/mean/StatefulPartitionedCallв,encoder1/fcn2/mean/StatefulPartitionedCall_1в%encoder1/fcn3/StatefulPartitionedCallв*encoder1/fcn3_gene/StatefulPartitionedCallв,encoder1/fcn3_gene/StatefulPartitionedCall_1в%encoder2/fcn1/StatefulPartitionedCallв'encoder2/fcn1/StatefulPartitionedCall_1в*encoder2/fcn2/mean/StatefulPartitionedCallв,encoder2/fcn2/mean/StatefulPartitionedCall_1в%encoder2/fcn3/StatefulPartitionedCall╖
%encoder2/fcn1/StatefulPartitionedCallStatefulPartitionedCallinput_2encoder2_fcn1_40050552encoder2_fcn1_40050554*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         x*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *T
fORM
K__inference_encoder2/fcn1_layer_call_and_return_conditional_losses_400494682'
%encoder2/fcn1/StatefulPartitionedCallў
*encoder2/fcn2/mean/StatefulPartitionedCallStatefulPartitionedCall.encoder2/fcn1/StatefulPartitionedCall:output:0encoder2_fcn2_mean_40050557encoder2_fcn2_mean_40050559*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         `*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *Y
fTRR
P__inference_encoder2/fcn2/mean_layer_call_and_return_conditional_losses_400494852,
*encoder2/fcn2/mean/StatefulPartitionedCall╨
*encoder1/fcn3_gene/StatefulPartitionedCallStatefulPartitionedCallinput_4encoder1_fcn3_gene_40050562encoder1_fcn3_gene_40050564*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         d*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *Y
fTRR
P__inference_encoder1/fcn3_gene_layer_call_and_return_conditional_losses_400495012,
*encoder1/fcn3_gene/StatefulPartitionedCall╖
%encoder1/fcn1/StatefulPartitionedCallStatefulPartitionedCallinput_3encoder1_fcn1_40050567encoder1_fcn1_40050569*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         d*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *T
fORM
K__inference_encoder1/fcn1_layer_call_and_return_conditional_losses_400495182'
%encoder1/fcn1/StatefulPartitionedCall╘
,encoder1/fcn3_gene/StatefulPartitionedCall_1StatefulPartitionedCallinput_6encoder1_fcn3_gene_40050562encoder1_fcn3_gene_40050564*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         d*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *Y
fTRR
P__inference_encoder1/fcn3_gene_layer_call_and_return_conditional_losses_400495012.
,encoder1/fcn3_gene/StatefulPartitionedCall_1╗
'encoder1/fcn1/StatefulPartitionedCall_1StatefulPartitionedCallinput_1encoder1_fcn1_40050567encoder1_fcn1_40050569*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         d*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *T
fORM
K__inference_encoder1/fcn1_layer_call_and_return_conditional_losses_400495182)
'encoder1/fcn1/StatefulPartitionedCall_1г
tf.math.square_5/SquareSquare3encoder2/fcn2/mean/StatefulPartitionedCall:output:0*
T0*'
_output_shapes
:         `2
tf.math.square_5/Squareг
tf.math.square_4/SquareSquare3encoder2/fcn2/mean/StatefulPartitionedCall:output:0*
T0*'
_output_shapes
:         `2
tf.math.square_4/Squarex
tf.math.square_3/SquareSquareinput_2*
T0*(
_output_shapes
:         А2
tf.math.square_3/Squarex
tf.math.square_2/SquareSquareinput_2*
T0*(
_output_shapes
:         А2
tf.math.square_2/Square╬
tf.math.subtract_3/SubSub.encoder1/fcn1/StatefulPartitionedCall:output:03encoder1/fcn3_gene/StatefulPartitionedCall:output:0*
T0*'
_output_shapes
:         d2
tf.math.subtract_3/Sub╬
tf.math.subtract/SubSub0encoder1/fcn1/StatefulPartitionedCall_1:output:05encoder1/fcn3_gene/StatefulPartitionedCall_1:output:0*
T0*'
_output_shapes
:         d2
tf.math.subtract/Subг
*tf.math.reduce_sum_3/Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
valueB :
         2,
*tf.math.reduce_sum_3/Sum/reduction_indices╨
tf.math.reduce_sum_3/SumSumtf.math.square_5/Square:y:03tf.math.reduce_sum_3/Sum/reduction_indices:output:0*
T0*'
_output_shapes
:         *
	keep_dims(2
tf.math.reduce_sum_3/Sumг
*tf.math.reduce_sum_2/Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
valueB :
         2,
*tf.math.reduce_sum_2/Sum/reduction_indices╨
tf.math.reduce_sum_2/SumSumtf.math.square_4/Square:y:03tf.math.reduce_sum_2/Sum/reduction_indices:output:0*
T0*'
_output_shapes
:         *
	keep_dims(2
tf.math.reduce_sum_2/Sumг
*tf.math.reduce_sum_1/Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
valueB :
         2,
*tf.math.reduce_sum_1/Sum/reduction_indices╨
tf.math.reduce_sum_1/SumSumtf.math.square_3/Square:y:03tf.math.reduce_sum_1/Sum/reduction_indices:output:0*
T0*'
_output_shapes
:         *
	keep_dims(2
tf.math.reduce_sum_1/SumЯ
(tf.math.reduce_sum/Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
valueB :
         2*
(tf.math.reduce_sum/Sum/reduction_indices╩
tf.math.reduce_sum/SumSumtf.math.square_2/Square:y:01tf.math.reduce_sum/Sum/reduction_indices:output:0*
T0*'
_output_shapes
:         *
	keep_dims(2
tf.math.reduce_sum/Sumу
*encoder1/fcn2/mean/StatefulPartitionedCallStatefulPartitionedCalltf.math.subtract_3/Sub:z:0encoder1_fcn2_mean_40050592encoder1_fcn2_mean_40050594*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         d*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *Y
fTRR
P__inference_encoder1/fcn2/mean_layer_call_and_return_conditional_losses_400495552,
*encoder1/fcn2/mean/StatefulPartitionedCallх
,encoder1/fcn2/mean/StatefulPartitionedCall_1StatefulPartitionedCalltf.math.subtract/Sub:z:0encoder1_fcn2_mean_40050592encoder1_fcn2_mean_40050594*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         d*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *Y
fTRR
P__inference_encoder1/fcn2/mean_layer_call_and_return_conditional_losses_400495552.
,encoder1/fcn2/mean/StatefulPartitionedCall_1З
tf.math.sqrt_2/SqrtSqrt!tf.math.reduce_sum_2/Sum:output:0*
T0*'
_output_shapes
:         2
tf.math.sqrt_2/SqrtЗ
tf.math.sqrt_3/SqrtSqrt!tf.math.reduce_sum_3/Sum:output:0*
T0*'
_output_shapes
:         2
tf.math.sqrt_3/SqrtБ
tf.math.sqrt/SqrtSqrttf.math.reduce_sum/Sum:output:0*
T0*'
_output_shapes
:         2
tf.math.sqrt/SqrtЗ
tf.math.sqrt_1/SqrtSqrt!tf.math.reduce_sum_1/Sum:output:0*
T0*'
_output_shapes
:         2
tf.math.sqrt_1/Sqrt╗
'encoder2/fcn1/StatefulPartitionedCall_1StatefulPartitionedCallinput_5encoder2_fcn1_40050552encoder2_fcn1_40050554*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         x*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *T
fORM
K__inference_encoder2/fcn1_layer_call_and_return_conditional_losses_400494682)
'encoder2/fcn1/StatefulPartitionedCall_1Б
+decoder1/layer_gene/StatefulPartitionedCallStatefulPartitionedCall3encoder1/fcn2/mean/StatefulPartitionedCall:output:0decoder1_layer_gene_40050607decoder1_layer_gene_40050609*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         d*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *Z
fURS
Q__inference_decoder1/layer_gene_layer_call_and_return_conditional_losses_400495822-
+decoder1/layer_gene/StatefulPartitionedCallЗ
-decoder1/layer_gene/StatefulPartitionedCall_1StatefulPartitionedCall5encoder1/fcn2/mean/StatefulPartitionedCall_1:output:0decoder1_layer_gene_40050607decoder1_layer_gene_40050609*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         d*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *Z
fURS
Q__inference_decoder1/layer_gene_layer_call_and_return_conditional_losses_400495822/
-decoder1/layer_gene/StatefulPartitionedCall_1°
tf.linalg.matmul_1/MatMulMatMul3encoder2/fcn2/mean/StatefulPartitionedCall:output:03encoder2/fcn2/mean/StatefulPartitionedCall:output:0*
T0*0
_output_shapes
:                  *
transpose_b(2
tf.linalg.matmul_1/MatMulЫ
tf.math.multiply_3/MulMultf.math.sqrt_2/Sqrt:y:0tf.math.sqrt_3/Sqrt:y:0*
T0*'
_output_shapes
:         2
tf.math.multiply_3/MulЬ
tf.linalg.matmul/MatMulMatMulinput_2input_2*
T0*0
_output_shapes
:                  *
transpose_b(2
tf.linalg.matmul/MatMulЩ
tf.math.multiply_2/MulMultf.math.sqrt/Sqrt:y:0tf.math.sqrt_1/Sqrt:y:0*
T0*'
_output_shapes
:         2
tf.math.multiply_2/Mul¤
,encoder2/fcn2/mean/StatefulPartitionedCall_1StatefulPartitionedCall0encoder2/fcn1/StatefulPartitionedCall_1:output:0encoder2_fcn2_mean_40050557encoder2_fcn2_mean_40050559*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         `*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *Y
fTRR
P__inference_encoder2/fcn2/mean_layer_call_and_return_conditional_losses_400494852.
,encoder2/fcn2/mean/StatefulPartitionedCall_1т
tf.__operators__.add_2/AddV2AddV24decoder1/layer_gene/StatefulPartitionedCall:output:03encoder1/fcn3_gene/StatefulPartitionedCall:output:0*
T0*'
_output_shapes
:         d2
tf.__operators__.add_2/AddV2т
tf.__operators__.add/AddV2AddV26decoder1/layer_gene/StatefulPartitionedCall_1:output:05encoder1/fcn3_gene/StatefulPartitionedCall_1:output:0*
T0*'
_output_shapes
:         d2
tf.__operators__.add/AddV2╜
tf.math.truediv_1/truedivRealDiv#tf.linalg.matmul_1/MatMul:product:0tf.math.multiply_3/Mul:z:0*
T0*0
_output_shapes
:                  2
tf.math.truediv_1/truediv╖
tf.math.truediv/truedivRealDiv!tf.linalg.matmul/MatMul:product:0tf.math.multiply_2/Mul:z:0*
T0*0
_output_shapes
:                  2
tf.math.truediv/truedivц
%decoder2/fcn1/StatefulPartitionedCallStatefulPartitionedCall5encoder2/fcn2/mean/StatefulPartitionedCall_1:output:0decoder2_fcn1_40050626decoder2_fcn1_40050628*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         А	*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *T
fORM
K__inference_decoder2/fcn1_layer_call_and_return_conditional_losses_400496132'
%decoder2/fcn1/StatefulPartitionedCall╤
%decoder1/fcn1/StatefulPartitionedCallStatefulPartitionedCall tf.__operators__.add_2/AddV2:z:0decoder1_fcn1_40050631decoder1_fcn1_40050633*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         А	*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *T
fORM
K__inference_decoder1/fcn1_layer_call_and_return_conditional_losses_400496302'
%decoder1/fcn1/StatefulPartitionedCallш
'decoder2/fcn1/StatefulPartitionedCall_1StatefulPartitionedCall3encoder2/fcn2/mean/StatefulPartitionedCall:output:0decoder2_fcn1_40050626decoder2_fcn1_40050628*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         А	*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *T
fORM
K__inference_decoder2/fcn1_layer_call_and_return_conditional_losses_400496132)
'decoder2/fcn1/StatefulPartitionedCall_1╙
'decoder1/fcn1/StatefulPartitionedCall_1StatefulPartitionedCalltf.__operators__.add/AddV2:z:0decoder1_fcn1_40050631decoder1_fcn1_40050633*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         А	*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *T
fORM
K__inference_decoder1/fcn1_layer_call_and_return_conditional_losses_400496302)
'decoder1/fcn1/StatefulPartitionedCall_1Ч
tf.nn.softmax_1/SoftmaxSoftmaxtf.math.truediv_1/truediv:z:0*
T0*0
_output_shapes
:                  2
tf.nn.softmax_1/SoftmaxС
tf.nn.softmax/SoftmaxSoftmaxtf.math.truediv/truediv:z:0*
T0*0
_output_shapes
:                  2
tf.nn.softmax/Softmaxy
tf.math.multiply_9/Mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *  HB2
tf.math.multiply_9/Mul/yЦ
tf.math.multiply_9/MulMulinput_5!tf.math.multiply_9/Mul/y:output:0*
T0*(
_output_shapes
:         А2
tf.math.multiply_9/Mul▀
%decoder2/fcn2/StatefulPartitionedCallStatefulPartitionedCall.decoder2/fcn1/StatefulPartitionedCall:output:0decoder2_fcn2_40050646decoder2_fcn2_40050648*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         А*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *T
fORM
K__inference_decoder2/fcn2_layer_call_and_return_conditional_losses_400496562'
%decoder2/fcn2/StatefulPartitionedCally
tf.math.multiply_8/Mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *  HB2
tf.math.multiply_8/Mul/yЦ
tf.math.multiply_8/MulMulinput_3!tf.math.multiply_8/Mul/y:output:0*
T0*(
_output_shapes
:         А2
tf.math.multiply_8/Mul▀
%decoder1/fcn2/StatefulPartitionedCallStatefulPartitionedCall.decoder1/fcn1/StatefulPartitionedCall:output:0decoder1_fcn2_40050653decoder1_fcn2_40050655*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         А*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *T
fORM
K__inference_decoder1/fcn2_layer_call_and_return_conditional_losses_400496742'
%decoder1/fcn2/StatefulPartitionedCallх
'decoder2/fcn2/StatefulPartitionedCall_1StatefulPartitionedCall0decoder2/fcn1/StatefulPartitionedCall_1:output:0decoder2_fcn2_40050646decoder2_fcn2_40050648*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         А*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *T
fORM
K__inference_decoder2/fcn2_layer_call_and_return_conditional_losses_400496562)
'decoder2/fcn2/StatefulPartitionedCall_1х
'decoder1/fcn2/StatefulPartitionedCall_1StatefulPartitionedCall0decoder1/fcn1/StatefulPartitionedCall_1:output:0decoder1_fcn2_40050653decoder1_fcn2_40050655*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         А*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *T
fORM
K__inference_decoder1/fcn2_layer_call_and_return_conditional_losses_400496742)
'decoder1/fcn2/StatefulPartitionedCall_1└
tf.math.truediv_3/truedivRealDiv!tf.nn.softmax_1/Softmax:softmax:0tf.nn.softmax/Softmax:softmax:0*
T0*0
_output_shapes
:                  2
tf.math.truediv_3/truediv└
tf.math.truediv_2/truedivRealDivtf.nn.softmax/Softmax:softmax:0!tf.nn.softmax_1/Softmax:softmax:0*
T0*0
_output_shapes
:                  2
tf.math.truediv_2/truediv╢
tf.math.subtract_5/SubSubtf.math.multiply_9/Mul:z:0.decoder2/fcn2/StatefulPartitionedCall:output:0*
T0*(
_output_shapes
:         А2
tf.math.subtract_5/Sub╢
tf.math.subtract_4/SubSubtf.math.multiply_8/Mul:z:0.decoder1/fcn2/StatefulPartitionedCall:output:0*
T0*(
_output_shapes
:         А2
tf.math.subtract_4/Suby
tf.math.multiply_1/Mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *  HB2
tf.math.multiply_1/Mul/yЦ
tf.math.multiply_1/MulMulinput_2!tf.math.multiply_1/Mul/y:output:0*
T0*(
_output_shapes
:         А2
tf.math.multiply_1/Mulu
tf.math.multiply/Mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *  HB2
tf.math.multiply/Mul/yР
tf.math.multiply/MulMulinput_1tf.math.multiply/Mul/y:output:0*
T0*(
_output_shapes
:         А2
tf.math.multiply/MulЗ
tf.math.log_1/LogLogtf.math.truediv_3/truediv:z:0*
T0*0
_output_shapes
:                  2
tf.math.log_1/LogГ
tf.math.log/LogLogtf.math.truediv_2/truediv:z:0*
T0*0
_output_shapes
:                  2
tf.math.log/LogЛ
tf.math.square_9/SquareSquaretf.math.subtract_5/Sub:z:0*
T0*(
_output_shapes
:         А2
tf.math.square_9/SquareЛ
tf.math.square_8/SquareSquaretf.math.subtract_4/Sub:z:0*
T0*(
_output_shapes
:         А2
tf.math.square_8/Square╕
tf.math.subtract_2/SubSubtf.math.multiply_1/Mul:z:00decoder2/fcn2/StatefulPartitionedCall_1:output:0*
T0*(
_output_shapes
:         А2
tf.math.subtract_2/Sub╢
tf.math.subtract_1/SubSubtf.math.multiply/Mul:z:00decoder1/fcn2/StatefulPartitionedCall_1:output:0*
T0*(
_output_shapes
:         А2
tf.math.subtract_1/Subм
tf.math.multiply_5/MulMul!tf.nn.softmax_1/Softmax:softmax:0tf.math.log_1/Log:y:0*
T0*0
_output_shapes
:                  2
tf.math.multiply_5/Mulи
tf.math.multiply_4/MulMultf.nn.softmax/Softmax:softmax:0tf.math.log/Log:y:0*
T0*0
_output_shapes
:                  2
tf.math.multiply_4/MulЛ
tf.math.reduce_mean_3/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2
tf.math.reduce_mean_3/Constд
tf.math.reduce_mean_3/MeanMeantf.math.square_8/Square:y:0$tf.math.reduce_mean_3/Const:output:0*
T0*
_output_shapes
: 2
tf.math.reduce_mean_3/MeanЛ
tf.math.reduce_mean_4/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2
tf.math.reduce_mean_4/Constд
tf.math.reduce_mean_4/MeanMeantf.math.square_9/Square:y:0$tf.math.reduce_mean_4/Const:output:0*
T0*
_output_shapes
: 2
tf.math.reduce_mean_4/MeanЛ
tf.math.square_1/SquareSquaretf.math.subtract_2/Sub:z:0*
T0*(
_output_shapes
:         А2
tf.math.square_1/SquareЗ
tf.math.square/SquareSquaretf.math.subtract_1/Sub:z:0*
T0*(
_output_shapes
:         А2
tf.math.square/Squareу
%encoder2/fcn3/StatefulPartitionedCallStatefulPartitionedCall3encoder2/fcn2/mean/StatefulPartitionedCall:output:0encoder2_fcn3_40050686encoder2_fcn3_40050688*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         P*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *T
fORM
K__inference_encoder2/fcn3_layer_call_and_return_conditional_losses_400497182'
%encoder2/fcn3/StatefulPartitionedCallх
%encoder1/fcn3/StatefulPartitionedCallStatefulPartitionedCall5encoder1/fcn2/mean/StatefulPartitionedCall_1:output:0encoder1_fcn3_40050691encoder1_fcn3_40050693*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         P*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *T
fORM
K__inference_encoder1/fcn3_layer_call_and_return_conditional_losses_400497342'
%encoder1/fcn3/StatefulPartitionedCallг
*tf.math.reduce_sum_4/Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
valueB :
         2,
*tf.math.reduce_sum_4/Sum/reduction_indices║
tf.math.reduce_sum_4/SumSumtf.math.multiply_4/Mul:z:03tf.math.reduce_sum_4/Sum/reduction_indices:output:0*
T0*#
_output_shapes
:         2
tf.math.reduce_sum_4/Sumг
*tf.math.reduce_sum_5/Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
valueB :
         2,
*tf.math.reduce_sum_5/Sum/reduction_indices║
tf.math.reduce_sum_5/SumSumtf.math.multiply_5/Mul:z:03tf.math.reduce_sum_5/Sum/reduction_indices:output:0*
T0*#
_output_shapes
:         2
tf.math.reduce_sum_5/Sum▓
tf.__operators__.add_16/AddV2AddV2#tf.math.reduce_mean_3/Mean:output:0#tf.math.reduce_mean_4/Mean:output:0*
T0*
_output_shapes
: 2
tf.__operators__.add_16/AddV2З
tf.math.reduce_mean/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2
tf.math.reduce_mean/ConstЬ
tf.math.reduce_mean/MeanMeantf.math.square/Square:y:0"tf.math.reduce_mean/Const:output:0*
T0*
_output_shapes
: 2
tf.math.reduce_mean/MeanЛ
tf.math.reduce_mean_1/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2
tf.math.reduce_mean_1/Constд
tf.math.reduce_mean_1/MeanMeantf.math.square_1/Square:y:0$tf.math.reduce_mean_1/Const:output:0*
T0*
_output_shapes
: 2
tf.math.reduce_mean_1/MeanЮ
tf.math.square_7/SquareSquare.encoder2/fcn3/StatefulPartitionedCall:output:0*
T0*'
_output_shapes
:         P2
tf.math.square_7/SquareЮ
tf.math.square_6/SquareSquare.encoder1/fcn3/StatefulPartitionedCall:output:0*
T0*'
_output_shapes
:         P2
tf.math.square_6/Square╣
tf.__operators__.add_1/AddV2AddV2!tf.math.reduce_sum_4/Sum:output:0!tf.math.reduce_sum_5/Sum:output:0*
T0*#
_output_shapes
:         2
tf.__operators__.add_1/AddV2Ж
tf.math.multiply_22/MulMulunknown!tf.__operators__.add_16/AddV2:z:0*
T0*
_output_shapes
: 2
tf.math.multiply_22/Mul░
tf.__operators__.add_15/AddV2AddV2!tf.math.reduce_mean/Mean:output:0#tf.math.reduce_mean_1/Mean:output:0*
T0*
_output_shapes
: 2
tf.__operators__.add_15/AddV2г
*tf.math.reduce_sum_8/Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
valueB :
         2,
*tf.math.reduce_sum_8/Sum/reduction_indices╗
tf.math.reduce_sum_8/SumSumtf.math.square_7/Square:y:03tf.math.reduce_sum_8/Sum/reduction_indices:output:0*
T0*#
_output_shapes
:         2
tf.math.reduce_sum_8/Sumг
*tf.math.reduce_sum_7/Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
valueB :
         2,
*tf.math.reduce_sum_7/Sum/reduction_indices╗
tf.math.reduce_sum_7/SumSumtf.math.square_6/Square:y:03tf.math.reduce_sum_7/Sum/reduction_indices:output:0*
T0*#
_output_shapes
:         2
tf.math.reduce_sum_7/SumД
tf.math.reduce_mean_2/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2
tf.math.reduce_mean_2/Constй
tf.math.reduce_mean_2/MeanMean tf.__operators__.add_1/AddV2:z:0$tf.math.reduce_mean_2/Const:output:0*
T0*
_output_shapes
: 2
tf.math.reduce_mean_2/MeanБ
tf.math.truediv_21/truediv/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
tf.math.truediv_21/truediv/yо
tf.math.truediv_21/truedivRealDiv!tf.__operators__.add_15/AddV2:z:0%tf.math.truediv_21/truediv/y:output:0*
T0*
_output_shapes
: 2
tf.math.truediv_21/truedivБ
tf.math.truediv_22/truediv/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
tf.math.truediv_22/truediv/yи
tf.math.truediv_22/truedivRealDivtf.math.multiply_22/Mul:z:0%tf.math.truediv_22/truediv/y:output:0*
T0*
_output_shapes
: 2
tf.math.truediv_22/truedivГ
tf.math.sqrt_4/SqrtSqrt!tf.math.reduce_sum_7/Sum:output:0*
T0*#
_output_shapes
:         2
tf.math.sqrt_4/SqrtГ
tf.math.sqrt_5/SqrtSqrt!tf.math.reduce_sum_8/Sum:output:0*
T0*#
_output_shapes
:         2
tf.math.sqrt_5/Sqrt╔
tf.math.multiply_6/MulMul.encoder1/fcn3/StatefulPartitionedCall:output:0.encoder2/fcn3/StatefulPartitionedCall:output:0*
T0*'
_output_shapes
:         P2
tf.math.multiply_6/Mulи
tf.__operators__.add_17/AddV2AddV2tf.math.truediv_21/truediv:z:0tf.math.truediv_22/truediv:z:0*
T0*
_output_shapes
: 2
tf.__operators__.add_17/AddV2{
tf.math.multiply_23/Mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2
tf.math.multiply_23/Mul/yг
tf.math.multiply_23/MulMul#tf.math.reduce_mean_2/Mean:output:0"tf.math.multiply_23/Mul/y:output:0*
T0*
_output_shapes
: 2
tf.math.multiply_23/Mulг
*tf.math.reduce_sum_6/Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
valueB :
         2,
*tf.math.reduce_sum_6/Sum/reduction_indices║
tf.math.reduce_sum_6/SumSumtf.math.multiply_6/Mul:z:03tf.math.reduce_sum_6/Sum/reduction_indices:output:0*
T0*#
_output_shapes
:         2
tf.math.reduce_sum_6/SumЧ
tf.math.multiply_7/MulMultf.math.sqrt_4/Sqrt:y:0tf.math.sqrt_5/Sqrt:y:0*
T0*#
_output_shapes
:         2
tf.math.multiply_7/Mulи
tf.__operators__.add_18/AddV2AddV2!tf.__operators__.add_17/AddV2:z:0tf.math.multiply_23/Mul:z:0*
T0*
_output_shapes
: 2
tf.__operators__.add_18/AddV2о
tf.math.truediv_4/truedivRealDiv!tf.math.reduce_sum_6/Sum:output:0tf.math.multiply_7/Mul:z:0*
T0*#
_output_shapes
:         2
tf.math.truediv_4/truedivЫ
IdentityIdentitytf.math.truediv_4/truediv:z:0&^decoder1/fcn1/StatefulPartitionedCall(^decoder1/fcn1/StatefulPartitionedCall_1&^decoder1/fcn2/StatefulPartitionedCall(^decoder1/fcn2/StatefulPartitionedCall_1,^decoder1/layer_gene/StatefulPartitionedCall.^decoder1/layer_gene/StatefulPartitionedCall_1&^decoder2/fcn1/StatefulPartitionedCall(^decoder2/fcn1/StatefulPartitionedCall_1&^decoder2/fcn2/StatefulPartitionedCall(^decoder2/fcn2/StatefulPartitionedCall_1&^encoder1/fcn1/StatefulPartitionedCall(^encoder1/fcn1/StatefulPartitionedCall_1+^encoder1/fcn2/mean/StatefulPartitionedCall-^encoder1/fcn2/mean/StatefulPartitionedCall_1&^encoder1/fcn3/StatefulPartitionedCall+^encoder1/fcn3_gene/StatefulPartitionedCall-^encoder1/fcn3_gene/StatefulPartitionedCall_1&^encoder2/fcn1/StatefulPartitionedCall(^encoder2/fcn1/StatefulPartitionedCall_1+^encoder2/fcn2/mean/StatefulPartitionedCall-^encoder2/fcn2/mean/StatefulPartitionedCall_1&^encoder2/fcn3/StatefulPartitionedCall*
T0*#
_output_shapes
:         2

IdentityЦ

Identity_1Identity!tf.__operators__.add_18/AddV2:z:0&^decoder1/fcn1/StatefulPartitionedCall(^decoder1/fcn1/StatefulPartitionedCall_1&^decoder1/fcn2/StatefulPartitionedCall(^decoder1/fcn2/StatefulPartitionedCall_1,^decoder1/layer_gene/StatefulPartitionedCall.^decoder1/layer_gene/StatefulPartitionedCall_1&^decoder2/fcn1/StatefulPartitionedCall(^decoder2/fcn1/StatefulPartitionedCall_1&^decoder2/fcn2/StatefulPartitionedCall(^decoder2/fcn2/StatefulPartitionedCall_1&^encoder1/fcn1/StatefulPartitionedCall(^encoder1/fcn1/StatefulPartitionedCall_1+^encoder1/fcn2/mean/StatefulPartitionedCall-^encoder1/fcn2/mean/StatefulPartitionedCall_1&^encoder1/fcn3/StatefulPartitionedCall+^encoder1/fcn3_gene/StatefulPartitionedCall-^encoder1/fcn3_gene/StatefulPartitionedCall_1&^encoder2/fcn1/StatefulPartitionedCall(^encoder2/fcn1/StatefulPartitionedCall_1+^encoder2/fcn2/mean/StatefulPartitionedCall-^encoder2/fcn2/mean/StatefulPartitionedCall_1&^encoder2/fcn3/StatefulPartitionedCall*
T0*
_output_shapes
: 2

Identity_1┤

Identity_2Identity.encoder1/fcn3/StatefulPartitionedCall:output:0&^decoder1/fcn1/StatefulPartitionedCall(^decoder1/fcn1/StatefulPartitionedCall_1&^decoder1/fcn2/StatefulPartitionedCall(^decoder1/fcn2/StatefulPartitionedCall_1,^decoder1/layer_gene/StatefulPartitionedCall.^decoder1/layer_gene/StatefulPartitionedCall_1&^decoder2/fcn1/StatefulPartitionedCall(^decoder2/fcn1/StatefulPartitionedCall_1&^decoder2/fcn2/StatefulPartitionedCall(^decoder2/fcn2/StatefulPartitionedCall_1&^encoder1/fcn1/StatefulPartitionedCall(^encoder1/fcn1/StatefulPartitionedCall_1+^encoder1/fcn2/mean/StatefulPartitionedCall-^encoder1/fcn2/mean/StatefulPartitionedCall_1&^encoder1/fcn3/StatefulPartitionedCall+^encoder1/fcn3_gene/StatefulPartitionedCall-^encoder1/fcn3_gene/StatefulPartitionedCall_1&^encoder2/fcn1/StatefulPartitionedCall(^encoder2/fcn1/StatefulPartitionedCall_1+^encoder2/fcn2/mean/StatefulPartitionedCall-^encoder2/fcn2/mean/StatefulPartitionedCall_1&^encoder2/fcn3/StatefulPartitionedCall*
T0*'
_output_shapes
:         P2

Identity_2┤

Identity_3Identity.encoder2/fcn3/StatefulPartitionedCall:output:0&^decoder1/fcn1/StatefulPartitionedCall(^decoder1/fcn1/StatefulPartitionedCall_1&^decoder1/fcn2/StatefulPartitionedCall(^decoder1/fcn2/StatefulPartitionedCall_1,^decoder1/layer_gene/StatefulPartitionedCall.^decoder1/layer_gene/StatefulPartitionedCall_1&^decoder2/fcn1/StatefulPartitionedCall(^decoder2/fcn1/StatefulPartitionedCall_1&^decoder2/fcn2/StatefulPartitionedCall(^decoder2/fcn2/StatefulPartitionedCall_1&^encoder1/fcn1/StatefulPartitionedCall(^encoder1/fcn1/StatefulPartitionedCall_1+^encoder1/fcn2/mean/StatefulPartitionedCall-^encoder1/fcn2/mean/StatefulPartitionedCall_1&^encoder1/fcn3/StatefulPartitionedCall+^encoder1/fcn3_gene/StatefulPartitionedCall-^encoder1/fcn3_gene/StatefulPartitionedCall_1&^encoder2/fcn1/StatefulPartitionedCall(^encoder2/fcn1/StatefulPartitionedCall_1+^encoder2/fcn2/mean/StatefulPartitionedCall-^encoder2/fcn2/mean/StatefulPartitionedCall_1&^encoder2/fcn3/StatefulPartitionedCall*
T0*'
_output_shapes
:         P2

Identity_3"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0*(
_construction_contextkEagerRuntime*┐
_input_shapesн
к:         А:         А:         А:         А:         А:         А: : : : : : : : : : : : : : : : : : : : : : : : : 2N
%decoder1/fcn1/StatefulPartitionedCall%decoder1/fcn1/StatefulPartitionedCall2R
'decoder1/fcn1/StatefulPartitionedCall_1'decoder1/fcn1/StatefulPartitionedCall_12N
%decoder1/fcn2/StatefulPartitionedCall%decoder1/fcn2/StatefulPartitionedCall2R
'decoder1/fcn2/StatefulPartitionedCall_1'decoder1/fcn2/StatefulPartitionedCall_12Z
+decoder1/layer_gene/StatefulPartitionedCall+decoder1/layer_gene/StatefulPartitionedCall2^
-decoder1/layer_gene/StatefulPartitionedCall_1-decoder1/layer_gene/StatefulPartitionedCall_12N
%decoder2/fcn1/StatefulPartitionedCall%decoder2/fcn1/StatefulPartitionedCall2R
'decoder2/fcn1/StatefulPartitionedCall_1'decoder2/fcn1/StatefulPartitionedCall_12N
%decoder2/fcn2/StatefulPartitionedCall%decoder2/fcn2/StatefulPartitionedCall2R
'decoder2/fcn2/StatefulPartitionedCall_1'decoder2/fcn2/StatefulPartitionedCall_12N
%encoder1/fcn1/StatefulPartitionedCall%encoder1/fcn1/StatefulPartitionedCall2R
'encoder1/fcn1/StatefulPartitionedCall_1'encoder1/fcn1/StatefulPartitionedCall_12X
*encoder1/fcn2/mean/StatefulPartitionedCall*encoder1/fcn2/mean/StatefulPartitionedCall2\
,encoder1/fcn2/mean/StatefulPartitionedCall_1,encoder1/fcn2/mean/StatefulPartitionedCall_12N
%encoder1/fcn3/StatefulPartitionedCall%encoder1/fcn3/StatefulPartitionedCall2X
*encoder1/fcn3_gene/StatefulPartitionedCall*encoder1/fcn3_gene/StatefulPartitionedCall2\
,encoder1/fcn3_gene/StatefulPartitionedCall_1,encoder1/fcn3_gene/StatefulPartitionedCall_12N
%encoder2/fcn1/StatefulPartitionedCall%encoder2/fcn1/StatefulPartitionedCall2R
'encoder2/fcn1/StatefulPartitionedCall_1'encoder2/fcn1/StatefulPartitionedCall_12X
*encoder2/fcn2/mean/StatefulPartitionedCall*encoder2/fcn2/mean/StatefulPartitionedCall2\
,encoder2/fcn2/mean/StatefulPartitionedCall_1,encoder2/fcn2/mean/StatefulPartitionedCall_12N
%encoder2/fcn3/StatefulPartitionedCall%encoder2/fcn3/StatefulPartitionedCall:Q M
(
_output_shapes
:         А
!
_user_specified_name	input_1:QM
(
_output_shapes
:         А
!
_user_specified_name	input_2:QM
(
_output_shapes
:         А
!
_user_specified_name	input_3:QM
(
_output_shapes
:         А
!
_user_specified_name	input_4:QM
(
_output_shapes
:         А
!
_user_specified_name	input_5:QM
(
_output_shapes
:         А
!
_user_specified_name	input_6:

_output_shapes
: 
╞

■
K__inference_decoder2/fcn1_layer_call_and_return_conditional_losses_40051545

inputs1
matmul_readvariableop_resource:	`А	.
biasadd_readvariableop_resource:	А	
identityИвBiasAdd/ReadVariableOpвMatMul/ReadVariableOpО
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	`А	*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         А	2
MatMulН
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:А	*
dtype02
BiasAdd/ReadVariableOpВ
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         А	2	
BiasAdd_
	LeakyRelu	LeakyReluBiasAdd:output:0*(
_output_shapes
:         А	2
	LeakyReluЭ
IdentityIdentityLeakyRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*(
_output_shapes
:         А	2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:         `: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:         `
 
_user_specified_nameinputs
т	
 
K__inference_decoder2/fcn2_layer_call_and_return_conditional_losses_40051583

inputs2
matmul_readvariableop_resource:
А	А.
biasadd_readvariableop_resource:	А
identityИвBiasAdd/ReadVariableOpвMatMul/ReadVariableOpП
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
А	А*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         А2
MatMulН
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:А*
dtype02
BiasAdd/ReadVariableOpВ
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         А2	
BiasAddЦ
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*(
_output_shapes
:         А2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:         А	: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:         А	
 
_user_specified_nameinputs
╫	
№
K__inference_encoder2/fcn3_layer_call_and_return_conditional_losses_40051621

inputs0
matmul_readvariableop_resource:`P-
biasadd_readvariableop_resource:P
identityИвBiasAdd/ReadVariableOpвMatMul/ReadVariableOpН
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:`P*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         P2
MatMulМ
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:P*
dtype02
BiasAdd/ReadVariableOpБ
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         P2	
BiasAddХ
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:         P2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:         `: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:         `
 
_user_specified_nameinputs
п
Ю
0__inference_encoder2/fcn1_layer_call_fn_40051415

inputs
unknown:	Аx
	unknown_0:x
identityИвStatefulPartitionedCall■
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         x*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *T
fORM
K__inference_encoder2/fcn1_layer_call_and_return_conditional_losses_400494682
StatefulPartitionedCallО
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:         x2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:         А: : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:         А
 
_user_specified_nameinputs
м
Э
0__inference_encoder2/fcn3_layer_call_fn_40051630

inputs
unknown:`P
	unknown_0:P
identityИвStatefulPartitionedCall■
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         P*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *T
fORM
K__inference_encoder2/fcn3_layer_call_and_return_conditional_losses_400497182
StatefulPartitionedCallО
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:         P2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:         `: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:         `
 
_user_specified_nameinputs
т	
 
K__inference_decoder1/fcn2_layer_call_and_return_conditional_losses_40049674

inputs2
matmul_readvariableop_resource:
А	А.
biasadd_readvariableop_resource:	А
identityИвBiasAdd/ReadVariableOpвMatMul/ReadVariableOpП
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
А	А*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         А2
MatMulН
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:А*
dtype02
BiasAdd/ReadVariableOpВ
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         А2	
BiasAddЦ
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*(
_output_shapes
:         А2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:         А	: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:         А	
 
_user_specified_nameinputs
├

Б
P__inference_encoder1/fcn2/mean_layer_call_and_return_conditional_losses_40051485

inputs0
matmul_readvariableop_resource:dd-
biasadd_readvariableop_resource:d
identityИвBiasAdd/ReadVariableOpвMatMul/ReadVariableOpН
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:dd*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         d2
MatMulМ
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:d*
dtype02
BiasAdd/ReadVariableOpБ
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         d2	
BiasAdd^
	LeakyRelu	LeakyReluBiasAdd:output:0*'
_output_shapes
:         d2
	LeakyReluЬ
IdentityIdentityLeakyRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:         d2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:         d: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:         d
 
_user_specified_nameinputs
╖

¤
K__inference_encoder1/fcn1_layer_call_and_return_conditional_losses_40051426

inputs1
matmul_readvariableop_resource:	Аd-
biasadd_readvariableop_resource:d
identityИвBiasAdd/ReadVariableOpвMatMul/ReadVariableOpО
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	Аd*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         d2
MatMulМ
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:d*
dtype02
BiasAdd/ReadVariableOpБ
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         d2	
BiasAddX
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:         d2
ReluЧ
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:         d2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:         А: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:         А
 
_user_specified_nameinputs
╫	
№
K__inference_encoder1/fcn3_layer_call_and_return_conditional_losses_40051602

inputs0
matmul_readvariableop_resource:dP-
biasadd_readvariableop_resource:P
identityИвBiasAdd/ReadVariableOpвMatMul/ReadVariableOpН
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:dP*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         P2
MatMulМ
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:P*
dtype02
BiasAdd/ReadVariableOpБ
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         P2	
BiasAddХ
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:         P2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:         d: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:         d
 
_user_specified_nameinputs
Д
Г
(__inference_model_layer_call_fn_40050351
input_1
input_2
input_3
input_4
input_5
input_6
unknown:	Аx
	unknown_0:x
	unknown_1:x`
	unknown_2:`
	unknown_3:	Аd
	unknown_4:d
	unknown_5:	Аd
	unknown_6:d
	unknown_7:dd
	unknown_8:d
	unknown_9:dd

unknown_10:d

unknown_11:	`А	

unknown_12:	А	

unknown_13:	dА	

unknown_14:	А	

unknown_15:
А	А

unknown_16:	А

unknown_17:
А	А

unknown_18:	А

unknown_19:`P

unknown_20:P

unknown_21:dP

unknown_22:P

unknown_23
identity

identity_1

identity_2

identity_3ИвStatefulPartitionedCallИ
StatefulPartitionedCallStatefulPartitionedCallinput_1input_2input_3input_4input_5input_6unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23**
Tin#
!2*
Tout
2*
_collective_manager_ids
 *K
_output_shapes9
7:         : :         P:         P*:
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8В *L
fGRE
C__inference_model_layer_call_and_return_conditional_losses_400502262
StatefulPartitionedCallК
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*#
_output_shapes
:         2

IdentityБ

Identity_1Identity StatefulPartitionedCall:output:1^StatefulPartitionedCall*
T0*
_output_shapes
: 2

Identity_1Т

Identity_2Identity StatefulPartitionedCall:output:2^StatefulPartitionedCall*
T0*'
_output_shapes
:         P2

Identity_2Т

Identity_3Identity StatefulPartitionedCall:output:3^StatefulPartitionedCall*
T0*'
_output_shapes
:         P2

Identity_3"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0*(
_construction_contextkEagerRuntime*┐
_input_shapesн
к:         А:         А:         А:         А:         А:         А: : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Q M
(
_output_shapes
:         А
!
_user_specified_name	input_1:QM
(
_output_shapes
:         А
!
_user_specified_name	input_2:QM
(
_output_shapes
:         А
!
_user_specified_name	input_3:QM
(
_output_shapes
:         А
!
_user_specified_name	input_4:QM
(
_output_shapes
:         А
!
_user_specified_name	input_5:QM
(
_output_shapes
:         А
!
_user_specified_name	input_6:

_output_shapes
: 
╫	
№
K__inference_encoder1/fcn3_layer_call_and_return_conditional_losses_40049734

inputs0
matmul_readvariableop_resource:dP-
biasadd_readvariableop_resource:P
identityИвBiasAdd/ReadVariableOpвMatMul/ReadVariableOpН
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:dP*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         P2
MatMulМ
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:P*
dtype02
BiasAdd/ReadVariableOpБ
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         P2	
BiasAddХ
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:         P2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:         d: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:         d
 
_user_specified_nameinputs
─

В
Q__inference_decoder1/layer_gene_layer_call_and_return_conditional_losses_40051505

inputs0
matmul_readvariableop_resource:dd-
biasadd_readvariableop_resource:d
identityИвBiasAdd/ReadVariableOpвMatMul/ReadVariableOpН
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:dd*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         d2
MatMulМ
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:d*
dtype02
BiasAdd/ReadVariableOpБ
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         d2	
BiasAdd^
	LeakyRelu	LeakyReluBiasAdd:output:0*'
_output_shapes
:         d2
	LeakyReluЬ
IdentityIdentityLeakyRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:         d2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:         d: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:         d
 
_user_specified_nameinputs
╞

■
K__inference_decoder1/fcn1_layer_call_and_return_conditional_losses_40051525

inputs1
matmul_readvariableop_resource:	dА	.
biasadd_readvariableop_resource:	А	
identityИвBiasAdd/ReadVariableOpвMatMul/ReadVariableOpО
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	dА	*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         А	2
MatMulН
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:А	*
dtype02
BiasAdd/ReadVariableOpВ
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         А	2	
BiasAdd_
	LeakyRelu	LeakyReluBiasAdd:output:0*(
_output_shapes
:         А	2
	LeakyReluЭ
IdentityIdentityLeakyRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*(
_output_shapes
:         А	2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:         d: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:         d
 
_user_specified_nameinputs
р	
В
P__inference_encoder1/fcn3_gene_layer_call_and_return_conditional_losses_40051445

inputs1
matmul_readvariableop_resource:	Аd-
biasadd_readvariableop_resource:d
identityИвBiasAdd/ReadVariableOpвMatMul/ReadVariableOpО
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	Аd*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         d2
MatMulМ
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:d*
dtype02
BiasAdd/ReadVariableOpБ
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         d2	
BiasAddХ
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:         d2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:         А: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:         А
 
_user_specified_nameinputs
╖

¤
K__inference_encoder2/fcn1_layer_call_and_return_conditional_losses_40049468

inputs1
matmul_readvariableop_resource:	Аx-
biasadd_readvariableop_resource:x
identityИвBiasAdd/ReadVariableOpвMatMul/ReadVariableOpО
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	Аx*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         x2
MatMulМ
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:x*
dtype02
BiasAdd/ReadVariableOpБ
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         x2	
BiasAddX
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:         x2
ReluЧ
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:         x2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:         А: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:         А
 
_user_specified_nameinputs
╢
в
5__inference_encoder1/fcn2/mean_layer_call_fn_40051494

inputs
unknown:dd
	unknown_0:d
identityИвStatefulPartitionedCallГ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         d*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *Y
fTRR
P__inference_encoder1/fcn2/mean_layer_call_and_return_conditional_losses_400495552
StatefulPartitionedCallО
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:         d2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:         d: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:         d
 
_user_specified_nameinputs
├

Б
P__inference_encoder2/fcn2/mean_layer_call_and_return_conditional_losses_40049485

inputs0
matmul_readvariableop_resource:x`-
biasadd_readvariableop_resource:`
identityИвBiasAdd/ReadVariableOpвMatMul/ReadVariableOpН
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:x`*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         `2
MatMulМ
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:`*
dtype02
BiasAdd/ReadVariableOpБ
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         `2	
BiasAdd^
	LeakyRelu	LeakyReluBiasAdd:output:0*'
_output_shapes
:         `2
	LeakyReluЬ
IdentityIdentityLeakyRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:         `2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:         x: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:         x
 
_user_specified_nameinputs
т
Б
&__inference_signature_wrapper_40050805
input_1
input_2
input_3
input_4
input_5
input_6
unknown:	Аx
	unknown_0:x
	unknown_1:x`
	unknown_2:`
	unknown_3:	Аd
	unknown_4:d
	unknown_5:	Аd
	unknown_6:d
	unknown_7:dd
	unknown_8:d
	unknown_9:dd

unknown_10:d

unknown_11:	`А	

unknown_12:	А	

unknown_13:	dА	

unknown_14:	А	

unknown_15:
А	А

unknown_16:	А

unknown_17:
А	А

unknown_18:	А

unknown_19:`P

unknown_20:P

unknown_21:dP

unknown_22:P

unknown_23
identity

identity_1

identity_2

identity_3ИвStatefulPartitionedCallш
StatefulPartitionedCallStatefulPartitionedCallinput_1input_2input_3input_4input_5input_6unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23**
Tin#
!2*
Tout
2*
_collective_manager_ids
 *K
_output_shapes9
7:         P:         P: :         *:
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8В *,
f'R%
#__inference__wrapped_model_400494402
StatefulPartitionedCallО
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:         P2

IdentityТ

Identity_1Identity StatefulPartitionedCall:output:1^StatefulPartitionedCall*
T0*'
_output_shapes
:         P2

Identity_1Б

Identity_2Identity StatefulPartitionedCall:output:2^StatefulPartitionedCall*
T0*
_output_shapes
: 2

Identity_2О

Identity_3Identity StatefulPartitionedCall:output:3^StatefulPartitionedCall*
T0*#
_output_shapes
:         2

Identity_3"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0*(
_construction_contextkEagerRuntime*┐
_input_shapesн
к:         А:         А:         А:         А:         А:         А: : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Q M
(
_output_shapes
:         А
!
_user_specified_name	input_1:QM
(
_output_shapes
:         А
!
_user_specified_name	input_2:QM
(
_output_shapes
:         А
!
_user_specified_name	input_3:QM
(
_output_shapes
:         А
!
_user_specified_name	input_4:QM
(
_output_shapes
:         А
!
_user_specified_name	input_5:QM
(
_output_shapes
:         А
!
_user_specified_name	input_6:

_output_shapes
: 
╖

¤
K__inference_encoder2/fcn1_layer_call_and_return_conditional_losses_40051406

inputs1
matmul_readvariableop_resource:	Аx-
biasadd_readvariableop_resource:x
identityИвBiasAdd/ReadVariableOpвMatMul/ReadVariableOpО
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	Аx*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         x2
MatMulМ
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:x*
dtype02
BiasAdd/ReadVariableOpБ
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         x2	
BiasAddX
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:         x2
ReluЧ
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:         x2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:         А: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:         А
 
_user_specified_nameinputs
╡Н
К
C__inference_model_layer_call_and_return_conditional_losses_40049781

inputs
inputs_1
inputs_2
inputs_3
inputs_4
inputs_5)
encoder2_fcn1_40049469:	Аx$
encoder2_fcn1_40049471:x-
encoder2_fcn2_mean_40049486:x`)
encoder2_fcn2_mean_40049488:`.
encoder1_fcn3_gene_40049502:	Аd)
encoder1_fcn3_gene_40049504:d)
encoder1_fcn1_40049519:	Аd$
encoder1_fcn1_40049521:d-
encoder1_fcn2_mean_40049556:dd)
encoder1_fcn2_mean_40049558:d.
decoder1_layer_gene_40049583:dd*
decoder1_layer_gene_40049585:d)
decoder2_fcn1_40049614:	`А	%
decoder2_fcn1_40049616:	А	)
decoder1_fcn1_40049631:	dА	%
decoder1_fcn1_40049633:	А	*
decoder2_fcn2_40049657:
А	А%
decoder2_fcn2_40049659:	А*
decoder1_fcn2_40049675:
А	А%
decoder1_fcn2_40049677:	А(
encoder2_fcn3_40049719:`P$
encoder2_fcn3_40049721:P(
encoder1_fcn3_40049735:dP$
encoder1_fcn3_40049737:P
unknown
identity

identity_1

identity_2

identity_3Ив%decoder1/fcn1/StatefulPartitionedCallв'decoder1/fcn1/StatefulPartitionedCall_1в%decoder1/fcn2/StatefulPartitionedCallв'decoder1/fcn2/StatefulPartitionedCall_1в+decoder1/layer_gene/StatefulPartitionedCallв-decoder1/layer_gene/StatefulPartitionedCall_1в%decoder2/fcn1/StatefulPartitionedCallв'decoder2/fcn1/StatefulPartitionedCall_1в%decoder2/fcn2/StatefulPartitionedCallв'decoder2/fcn2/StatefulPartitionedCall_1в%encoder1/fcn1/StatefulPartitionedCallв'encoder1/fcn1/StatefulPartitionedCall_1в*encoder1/fcn2/mean/StatefulPartitionedCallв,encoder1/fcn2/mean/StatefulPartitionedCall_1в%encoder1/fcn3/StatefulPartitionedCallв*encoder1/fcn3_gene/StatefulPartitionedCallв,encoder1/fcn3_gene/StatefulPartitionedCall_1в%encoder2/fcn1/StatefulPartitionedCallв'encoder2/fcn1/StatefulPartitionedCall_1в*encoder2/fcn2/mean/StatefulPartitionedCallв,encoder2/fcn2/mean/StatefulPartitionedCall_1в%encoder2/fcn3/StatefulPartitionedCall╕
%encoder2/fcn1/StatefulPartitionedCallStatefulPartitionedCallinputs_1encoder2_fcn1_40049469encoder2_fcn1_40049471*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         x*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *T
fORM
K__inference_encoder2/fcn1_layer_call_and_return_conditional_losses_400494682'
%encoder2/fcn1/StatefulPartitionedCallў
*encoder2/fcn2/mean/StatefulPartitionedCallStatefulPartitionedCall.encoder2/fcn1/StatefulPartitionedCall:output:0encoder2_fcn2_mean_40049486encoder2_fcn2_mean_40049488*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         `*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *Y
fTRR
P__inference_encoder2/fcn2/mean_layer_call_and_return_conditional_losses_400494852,
*encoder2/fcn2/mean/StatefulPartitionedCall╤
*encoder1/fcn3_gene/StatefulPartitionedCallStatefulPartitionedCallinputs_3encoder1_fcn3_gene_40049502encoder1_fcn3_gene_40049504*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         d*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *Y
fTRR
P__inference_encoder1/fcn3_gene_layer_call_and_return_conditional_losses_400495012,
*encoder1/fcn3_gene/StatefulPartitionedCall╕
%encoder1/fcn1/StatefulPartitionedCallStatefulPartitionedCallinputs_2encoder1_fcn1_40049519encoder1_fcn1_40049521*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         d*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *T
fORM
K__inference_encoder1/fcn1_layer_call_and_return_conditional_losses_400495182'
%encoder1/fcn1/StatefulPartitionedCall╒
,encoder1/fcn3_gene/StatefulPartitionedCall_1StatefulPartitionedCallinputs_5encoder1_fcn3_gene_40049502encoder1_fcn3_gene_40049504*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         d*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *Y
fTRR
P__inference_encoder1/fcn3_gene_layer_call_and_return_conditional_losses_400495012.
,encoder1/fcn3_gene/StatefulPartitionedCall_1║
'encoder1/fcn1/StatefulPartitionedCall_1StatefulPartitionedCallinputsencoder1_fcn1_40049519encoder1_fcn1_40049521*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         d*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *T
fORM
K__inference_encoder1/fcn1_layer_call_and_return_conditional_losses_400495182)
'encoder1/fcn1/StatefulPartitionedCall_1г
tf.math.square_5/SquareSquare3encoder2/fcn2/mean/StatefulPartitionedCall:output:0*
T0*'
_output_shapes
:         `2
tf.math.square_5/Squareг
tf.math.square_4/SquareSquare3encoder2/fcn2/mean/StatefulPartitionedCall:output:0*
T0*'
_output_shapes
:         `2
tf.math.square_4/Squarey
tf.math.square_3/SquareSquareinputs_1*
T0*(
_output_shapes
:         А2
tf.math.square_3/Squarey
tf.math.square_2/SquareSquareinputs_1*
T0*(
_output_shapes
:         А2
tf.math.square_2/Square╬
tf.math.subtract_3/SubSub.encoder1/fcn1/StatefulPartitionedCall:output:03encoder1/fcn3_gene/StatefulPartitionedCall:output:0*
T0*'
_output_shapes
:         d2
tf.math.subtract_3/Sub╬
tf.math.subtract/SubSub0encoder1/fcn1/StatefulPartitionedCall_1:output:05encoder1/fcn3_gene/StatefulPartitionedCall_1:output:0*
T0*'
_output_shapes
:         d2
tf.math.subtract/Subг
*tf.math.reduce_sum_3/Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
valueB :
         2,
*tf.math.reduce_sum_3/Sum/reduction_indices╨
tf.math.reduce_sum_3/SumSumtf.math.square_5/Square:y:03tf.math.reduce_sum_3/Sum/reduction_indices:output:0*
T0*'
_output_shapes
:         *
	keep_dims(2
tf.math.reduce_sum_3/Sumг
*tf.math.reduce_sum_2/Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
valueB :
         2,
*tf.math.reduce_sum_2/Sum/reduction_indices╨
tf.math.reduce_sum_2/SumSumtf.math.square_4/Square:y:03tf.math.reduce_sum_2/Sum/reduction_indices:output:0*
T0*'
_output_shapes
:         *
	keep_dims(2
tf.math.reduce_sum_2/Sumг
*tf.math.reduce_sum_1/Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
valueB :
         2,
*tf.math.reduce_sum_1/Sum/reduction_indices╨
tf.math.reduce_sum_1/SumSumtf.math.square_3/Square:y:03tf.math.reduce_sum_1/Sum/reduction_indices:output:0*
T0*'
_output_shapes
:         *
	keep_dims(2
tf.math.reduce_sum_1/SumЯ
(tf.math.reduce_sum/Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
valueB :
         2*
(tf.math.reduce_sum/Sum/reduction_indices╩
tf.math.reduce_sum/SumSumtf.math.square_2/Square:y:01tf.math.reduce_sum/Sum/reduction_indices:output:0*
T0*'
_output_shapes
:         *
	keep_dims(2
tf.math.reduce_sum/Sumу
*encoder1/fcn2/mean/StatefulPartitionedCallStatefulPartitionedCalltf.math.subtract_3/Sub:z:0encoder1_fcn2_mean_40049556encoder1_fcn2_mean_40049558*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         d*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *Y
fTRR
P__inference_encoder1/fcn2/mean_layer_call_and_return_conditional_losses_400495552,
*encoder1/fcn2/mean/StatefulPartitionedCallх
,encoder1/fcn2/mean/StatefulPartitionedCall_1StatefulPartitionedCalltf.math.subtract/Sub:z:0encoder1_fcn2_mean_40049556encoder1_fcn2_mean_40049558*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         d*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *Y
fTRR
P__inference_encoder1/fcn2/mean_layer_call_and_return_conditional_losses_400495552.
,encoder1/fcn2/mean/StatefulPartitionedCall_1З
tf.math.sqrt_2/SqrtSqrt!tf.math.reduce_sum_2/Sum:output:0*
T0*'
_output_shapes
:         2
tf.math.sqrt_2/SqrtЗ
tf.math.sqrt_3/SqrtSqrt!tf.math.reduce_sum_3/Sum:output:0*
T0*'
_output_shapes
:         2
tf.math.sqrt_3/SqrtБ
tf.math.sqrt/SqrtSqrttf.math.reduce_sum/Sum:output:0*
T0*'
_output_shapes
:         2
tf.math.sqrt/SqrtЗ
tf.math.sqrt_1/SqrtSqrt!tf.math.reduce_sum_1/Sum:output:0*
T0*'
_output_shapes
:         2
tf.math.sqrt_1/Sqrt╝
'encoder2/fcn1/StatefulPartitionedCall_1StatefulPartitionedCallinputs_4encoder2_fcn1_40049469encoder2_fcn1_40049471*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         x*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *T
fORM
K__inference_encoder2/fcn1_layer_call_and_return_conditional_losses_400494682)
'encoder2/fcn1/StatefulPartitionedCall_1Б
+decoder1/layer_gene/StatefulPartitionedCallStatefulPartitionedCall3encoder1/fcn2/mean/StatefulPartitionedCall:output:0decoder1_layer_gene_40049583decoder1_layer_gene_40049585*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         d*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *Z
fURS
Q__inference_decoder1/layer_gene_layer_call_and_return_conditional_losses_400495822-
+decoder1/layer_gene/StatefulPartitionedCallЗ
-decoder1/layer_gene/StatefulPartitionedCall_1StatefulPartitionedCall5encoder1/fcn2/mean/StatefulPartitionedCall_1:output:0decoder1_layer_gene_40049583decoder1_layer_gene_40049585*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         d*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *Z
fURS
Q__inference_decoder1/layer_gene_layer_call_and_return_conditional_losses_400495822/
-decoder1/layer_gene/StatefulPartitionedCall_1°
tf.linalg.matmul_1/MatMulMatMul3encoder2/fcn2/mean/StatefulPartitionedCall:output:03encoder2/fcn2/mean/StatefulPartitionedCall:output:0*
T0*0
_output_shapes
:                  *
transpose_b(2
tf.linalg.matmul_1/MatMulЫ
tf.math.multiply_3/MulMultf.math.sqrt_2/Sqrt:y:0tf.math.sqrt_3/Sqrt:y:0*
T0*'
_output_shapes
:         2
tf.math.multiply_3/MulЮ
tf.linalg.matmul/MatMulMatMulinputs_1inputs_1*
T0*0
_output_shapes
:                  *
transpose_b(2
tf.linalg.matmul/MatMulЩ
tf.math.multiply_2/MulMultf.math.sqrt/Sqrt:y:0tf.math.sqrt_1/Sqrt:y:0*
T0*'
_output_shapes
:         2
tf.math.multiply_2/Mul¤
,encoder2/fcn2/mean/StatefulPartitionedCall_1StatefulPartitionedCall0encoder2/fcn1/StatefulPartitionedCall_1:output:0encoder2_fcn2_mean_40049486encoder2_fcn2_mean_40049488*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         `*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *Y
fTRR
P__inference_encoder2/fcn2/mean_layer_call_and_return_conditional_losses_400494852.
,encoder2/fcn2/mean/StatefulPartitionedCall_1т
tf.__operators__.add_2/AddV2AddV24decoder1/layer_gene/StatefulPartitionedCall:output:03encoder1/fcn3_gene/StatefulPartitionedCall:output:0*
T0*'
_output_shapes
:         d2
tf.__operators__.add_2/AddV2т
tf.__operators__.add/AddV2AddV26decoder1/layer_gene/StatefulPartitionedCall_1:output:05encoder1/fcn3_gene/StatefulPartitionedCall_1:output:0*
T0*'
_output_shapes
:         d2
tf.__operators__.add/AddV2╜
tf.math.truediv_1/truedivRealDiv#tf.linalg.matmul_1/MatMul:product:0tf.math.multiply_3/Mul:z:0*
T0*0
_output_shapes
:                  2
tf.math.truediv_1/truediv╖
tf.math.truediv/truedivRealDiv!tf.linalg.matmul/MatMul:product:0tf.math.multiply_2/Mul:z:0*
T0*0
_output_shapes
:                  2
tf.math.truediv/truedivц
%decoder2/fcn1/StatefulPartitionedCallStatefulPartitionedCall5encoder2/fcn2/mean/StatefulPartitionedCall_1:output:0decoder2_fcn1_40049614decoder2_fcn1_40049616*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         А	*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *T
fORM
K__inference_decoder2/fcn1_layer_call_and_return_conditional_losses_400496132'
%decoder2/fcn1/StatefulPartitionedCall╤
%decoder1/fcn1/StatefulPartitionedCallStatefulPartitionedCall tf.__operators__.add_2/AddV2:z:0decoder1_fcn1_40049631decoder1_fcn1_40049633*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         А	*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *T
fORM
K__inference_decoder1/fcn1_layer_call_and_return_conditional_losses_400496302'
%decoder1/fcn1/StatefulPartitionedCallш
'decoder2/fcn1/StatefulPartitionedCall_1StatefulPartitionedCall3encoder2/fcn2/mean/StatefulPartitionedCall:output:0decoder2_fcn1_40049614decoder2_fcn1_40049616*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         А	*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *T
fORM
K__inference_decoder2/fcn1_layer_call_and_return_conditional_losses_400496132)
'decoder2/fcn1/StatefulPartitionedCall_1╙
'decoder1/fcn1/StatefulPartitionedCall_1StatefulPartitionedCalltf.__operators__.add/AddV2:z:0decoder1_fcn1_40049631decoder1_fcn1_40049633*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         А	*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *T
fORM
K__inference_decoder1/fcn1_layer_call_and_return_conditional_losses_400496302)
'decoder1/fcn1/StatefulPartitionedCall_1Ч
tf.nn.softmax_1/SoftmaxSoftmaxtf.math.truediv_1/truediv:z:0*
T0*0
_output_shapes
:                  2
tf.nn.softmax_1/SoftmaxС
tf.nn.softmax/SoftmaxSoftmaxtf.math.truediv/truediv:z:0*
T0*0
_output_shapes
:                  2
tf.nn.softmax/Softmaxy
tf.math.multiply_9/Mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *  HB2
tf.math.multiply_9/Mul/yЧ
tf.math.multiply_9/MulMulinputs_4!tf.math.multiply_9/Mul/y:output:0*
T0*(
_output_shapes
:         А2
tf.math.multiply_9/Mul▀
%decoder2/fcn2/StatefulPartitionedCallStatefulPartitionedCall.decoder2/fcn1/StatefulPartitionedCall:output:0decoder2_fcn2_40049657decoder2_fcn2_40049659*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         А*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *T
fORM
K__inference_decoder2/fcn2_layer_call_and_return_conditional_losses_400496562'
%decoder2/fcn2/StatefulPartitionedCally
tf.math.multiply_8/Mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *  HB2
tf.math.multiply_8/Mul/yЧ
tf.math.multiply_8/MulMulinputs_2!tf.math.multiply_8/Mul/y:output:0*
T0*(
_output_shapes
:         А2
tf.math.multiply_8/Mul▀
%decoder1/fcn2/StatefulPartitionedCallStatefulPartitionedCall.decoder1/fcn1/StatefulPartitionedCall:output:0decoder1_fcn2_40049675decoder1_fcn2_40049677*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         А*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *T
fORM
K__inference_decoder1/fcn2_layer_call_and_return_conditional_losses_400496742'
%decoder1/fcn2/StatefulPartitionedCallх
'decoder2/fcn2/StatefulPartitionedCall_1StatefulPartitionedCall0decoder2/fcn1/StatefulPartitionedCall_1:output:0decoder2_fcn2_40049657decoder2_fcn2_40049659*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         А*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *T
fORM
K__inference_decoder2/fcn2_layer_call_and_return_conditional_losses_400496562)
'decoder2/fcn2/StatefulPartitionedCall_1х
'decoder1/fcn2/StatefulPartitionedCall_1StatefulPartitionedCall0decoder1/fcn1/StatefulPartitionedCall_1:output:0decoder1_fcn2_40049675decoder1_fcn2_40049677*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         А*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *T
fORM
K__inference_decoder1/fcn2_layer_call_and_return_conditional_losses_400496742)
'decoder1/fcn2/StatefulPartitionedCall_1└
tf.math.truediv_3/truedivRealDiv!tf.nn.softmax_1/Softmax:softmax:0tf.nn.softmax/Softmax:softmax:0*
T0*0
_output_shapes
:                  2
tf.math.truediv_3/truediv└
tf.math.truediv_2/truedivRealDivtf.nn.softmax/Softmax:softmax:0!tf.nn.softmax_1/Softmax:softmax:0*
T0*0
_output_shapes
:                  2
tf.math.truediv_2/truediv╢
tf.math.subtract_5/SubSubtf.math.multiply_9/Mul:z:0.decoder2/fcn2/StatefulPartitionedCall:output:0*
T0*(
_output_shapes
:         А2
tf.math.subtract_5/Sub╢
tf.math.subtract_4/SubSubtf.math.multiply_8/Mul:z:0.decoder1/fcn2/StatefulPartitionedCall:output:0*
T0*(
_output_shapes
:         А2
tf.math.subtract_4/Suby
tf.math.multiply_1/Mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *  HB2
tf.math.multiply_1/Mul/yЧ
tf.math.multiply_1/MulMulinputs_1!tf.math.multiply_1/Mul/y:output:0*
T0*(
_output_shapes
:         А2
tf.math.multiply_1/Mulu
tf.math.multiply/Mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *  HB2
tf.math.multiply/Mul/yП
tf.math.multiply/MulMulinputstf.math.multiply/Mul/y:output:0*
T0*(
_output_shapes
:         А2
tf.math.multiply/MulЗ
tf.math.log_1/LogLogtf.math.truediv_3/truediv:z:0*
T0*0
_output_shapes
:                  2
tf.math.log_1/LogГ
tf.math.log/LogLogtf.math.truediv_2/truediv:z:0*
T0*0
_output_shapes
:                  2
tf.math.log/LogЛ
tf.math.square_9/SquareSquaretf.math.subtract_5/Sub:z:0*
T0*(
_output_shapes
:         А2
tf.math.square_9/SquareЛ
tf.math.square_8/SquareSquaretf.math.subtract_4/Sub:z:0*
T0*(
_output_shapes
:         А2
tf.math.square_8/Square╕
tf.math.subtract_2/SubSubtf.math.multiply_1/Mul:z:00decoder2/fcn2/StatefulPartitionedCall_1:output:0*
T0*(
_output_shapes
:         А2
tf.math.subtract_2/Sub╢
tf.math.subtract_1/SubSubtf.math.multiply/Mul:z:00decoder1/fcn2/StatefulPartitionedCall_1:output:0*
T0*(
_output_shapes
:         А2
tf.math.subtract_1/Subм
tf.math.multiply_5/MulMul!tf.nn.softmax_1/Softmax:softmax:0tf.math.log_1/Log:y:0*
T0*0
_output_shapes
:                  2
tf.math.multiply_5/Mulи
tf.math.multiply_4/MulMultf.nn.softmax/Softmax:softmax:0tf.math.log/Log:y:0*
T0*0
_output_shapes
:                  2
tf.math.multiply_4/MulЛ
tf.math.reduce_mean_3/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2
tf.math.reduce_mean_3/Constд
tf.math.reduce_mean_3/MeanMeantf.math.square_8/Square:y:0$tf.math.reduce_mean_3/Const:output:0*
T0*
_output_shapes
: 2
tf.math.reduce_mean_3/MeanЛ
tf.math.reduce_mean_4/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2
tf.math.reduce_mean_4/Constд
tf.math.reduce_mean_4/MeanMeantf.math.square_9/Square:y:0$tf.math.reduce_mean_4/Const:output:0*
T0*
_output_shapes
: 2
tf.math.reduce_mean_4/MeanЛ
tf.math.square_1/SquareSquaretf.math.subtract_2/Sub:z:0*
T0*(
_output_shapes
:         А2
tf.math.square_1/SquareЗ
tf.math.square/SquareSquaretf.math.subtract_1/Sub:z:0*
T0*(
_output_shapes
:         А2
tf.math.square/Squareу
%encoder2/fcn3/StatefulPartitionedCallStatefulPartitionedCall3encoder2/fcn2/mean/StatefulPartitionedCall:output:0encoder2_fcn3_40049719encoder2_fcn3_40049721*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         P*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *T
fORM
K__inference_encoder2/fcn3_layer_call_and_return_conditional_losses_400497182'
%encoder2/fcn3/StatefulPartitionedCallх
%encoder1/fcn3/StatefulPartitionedCallStatefulPartitionedCall5encoder1/fcn2/mean/StatefulPartitionedCall_1:output:0encoder1_fcn3_40049735encoder1_fcn3_40049737*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         P*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *T
fORM
K__inference_encoder1/fcn3_layer_call_and_return_conditional_losses_400497342'
%encoder1/fcn3/StatefulPartitionedCallг
*tf.math.reduce_sum_4/Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
valueB :
         2,
*tf.math.reduce_sum_4/Sum/reduction_indices║
tf.math.reduce_sum_4/SumSumtf.math.multiply_4/Mul:z:03tf.math.reduce_sum_4/Sum/reduction_indices:output:0*
T0*#
_output_shapes
:         2
tf.math.reduce_sum_4/Sumг
*tf.math.reduce_sum_5/Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
valueB :
         2,
*tf.math.reduce_sum_5/Sum/reduction_indices║
tf.math.reduce_sum_5/SumSumtf.math.multiply_5/Mul:z:03tf.math.reduce_sum_5/Sum/reduction_indices:output:0*
T0*#
_output_shapes
:         2
tf.math.reduce_sum_5/Sum▓
tf.__operators__.add_16/AddV2AddV2#tf.math.reduce_mean_3/Mean:output:0#tf.math.reduce_mean_4/Mean:output:0*
T0*
_output_shapes
: 2
tf.__operators__.add_16/AddV2З
tf.math.reduce_mean/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2
tf.math.reduce_mean/ConstЬ
tf.math.reduce_mean/MeanMeantf.math.square/Square:y:0"tf.math.reduce_mean/Const:output:0*
T0*
_output_shapes
: 2
tf.math.reduce_mean/MeanЛ
tf.math.reduce_mean_1/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2
tf.math.reduce_mean_1/Constд
tf.math.reduce_mean_1/MeanMeantf.math.square_1/Square:y:0$tf.math.reduce_mean_1/Const:output:0*
T0*
_output_shapes
: 2
tf.math.reduce_mean_1/MeanЮ
tf.math.square_7/SquareSquare.encoder2/fcn3/StatefulPartitionedCall:output:0*
T0*'
_output_shapes
:         P2
tf.math.square_7/SquareЮ
tf.math.square_6/SquareSquare.encoder1/fcn3/StatefulPartitionedCall:output:0*
T0*'
_output_shapes
:         P2
tf.math.square_6/Square╣
tf.__operators__.add_1/AddV2AddV2!tf.math.reduce_sum_4/Sum:output:0!tf.math.reduce_sum_5/Sum:output:0*
T0*#
_output_shapes
:         2
tf.__operators__.add_1/AddV2Ж
tf.math.multiply_22/MulMulunknown!tf.__operators__.add_16/AddV2:z:0*
T0*
_output_shapes
: 2
tf.math.multiply_22/Mul░
tf.__operators__.add_15/AddV2AddV2!tf.math.reduce_mean/Mean:output:0#tf.math.reduce_mean_1/Mean:output:0*
T0*
_output_shapes
: 2
tf.__operators__.add_15/AddV2г
*tf.math.reduce_sum_8/Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
valueB :
         2,
*tf.math.reduce_sum_8/Sum/reduction_indices╗
tf.math.reduce_sum_8/SumSumtf.math.square_7/Square:y:03tf.math.reduce_sum_8/Sum/reduction_indices:output:0*
T0*#
_output_shapes
:         2
tf.math.reduce_sum_8/Sumг
*tf.math.reduce_sum_7/Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
valueB :
         2,
*tf.math.reduce_sum_7/Sum/reduction_indices╗
tf.math.reduce_sum_7/SumSumtf.math.square_6/Square:y:03tf.math.reduce_sum_7/Sum/reduction_indices:output:0*
T0*#
_output_shapes
:         2
tf.math.reduce_sum_7/SumД
tf.math.reduce_mean_2/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2
tf.math.reduce_mean_2/Constй
tf.math.reduce_mean_2/MeanMean tf.__operators__.add_1/AddV2:z:0$tf.math.reduce_mean_2/Const:output:0*
T0*
_output_shapes
: 2
tf.math.reduce_mean_2/MeanБ
tf.math.truediv_21/truediv/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
tf.math.truediv_21/truediv/yо
tf.math.truediv_21/truedivRealDiv!tf.__operators__.add_15/AddV2:z:0%tf.math.truediv_21/truediv/y:output:0*
T0*
_output_shapes
: 2
tf.math.truediv_21/truedivБ
tf.math.truediv_22/truediv/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
tf.math.truediv_22/truediv/yи
tf.math.truediv_22/truedivRealDivtf.math.multiply_22/Mul:z:0%tf.math.truediv_22/truediv/y:output:0*
T0*
_output_shapes
: 2
tf.math.truediv_22/truedivГ
tf.math.sqrt_4/SqrtSqrt!tf.math.reduce_sum_7/Sum:output:0*
T0*#
_output_shapes
:         2
tf.math.sqrt_4/SqrtГ
tf.math.sqrt_5/SqrtSqrt!tf.math.reduce_sum_8/Sum:output:0*
T0*#
_output_shapes
:         2
tf.math.sqrt_5/Sqrt╔
tf.math.multiply_6/MulMul.encoder1/fcn3/StatefulPartitionedCall:output:0.encoder2/fcn3/StatefulPartitionedCall:output:0*
T0*'
_output_shapes
:         P2
tf.math.multiply_6/Mulи
tf.__operators__.add_17/AddV2AddV2tf.math.truediv_21/truediv:z:0tf.math.truediv_22/truediv:z:0*
T0*
_output_shapes
: 2
tf.__operators__.add_17/AddV2{
tf.math.multiply_23/Mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2
tf.math.multiply_23/Mul/yг
tf.math.multiply_23/MulMul#tf.math.reduce_mean_2/Mean:output:0"tf.math.multiply_23/Mul/y:output:0*
T0*
_output_shapes
: 2
tf.math.multiply_23/Mulг
*tf.math.reduce_sum_6/Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
valueB :
         2,
*tf.math.reduce_sum_6/Sum/reduction_indices║
tf.math.reduce_sum_6/SumSumtf.math.multiply_6/Mul:z:03tf.math.reduce_sum_6/Sum/reduction_indices:output:0*
T0*#
_output_shapes
:         2
tf.math.reduce_sum_6/SumЧ
tf.math.multiply_7/MulMultf.math.sqrt_4/Sqrt:y:0tf.math.sqrt_5/Sqrt:y:0*
T0*#
_output_shapes
:         2
tf.math.multiply_7/Mulи
tf.__operators__.add_18/AddV2AddV2!tf.__operators__.add_17/AddV2:z:0tf.math.multiply_23/Mul:z:0*
T0*
_output_shapes
: 2
tf.__operators__.add_18/AddV2о
tf.math.truediv_4/truedivRealDiv!tf.math.reduce_sum_6/Sum:output:0tf.math.multiply_7/Mul:z:0*
T0*#
_output_shapes
:         2
tf.math.truediv_4/truedivЫ
IdentityIdentitytf.math.truediv_4/truediv:z:0&^decoder1/fcn1/StatefulPartitionedCall(^decoder1/fcn1/StatefulPartitionedCall_1&^decoder1/fcn2/StatefulPartitionedCall(^decoder1/fcn2/StatefulPartitionedCall_1,^decoder1/layer_gene/StatefulPartitionedCall.^decoder1/layer_gene/StatefulPartitionedCall_1&^decoder2/fcn1/StatefulPartitionedCall(^decoder2/fcn1/StatefulPartitionedCall_1&^decoder2/fcn2/StatefulPartitionedCall(^decoder2/fcn2/StatefulPartitionedCall_1&^encoder1/fcn1/StatefulPartitionedCall(^encoder1/fcn1/StatefulPartitionedCall_1+^encoder1/fcn2/mean/StatefulPartitionedCall-^encoder1/fcn2/mean/StatefulPartitionedCall_1&^encoder1/fcn3/StatefulPartitionedCall+^encoder1/fcn3_gene/StatefulPartitionedCall-^encoder1/fcn3_gene/StatefulPartitionedCall_1&^encoder2/fcn1/StatefulPartitionedCall(^encoder2/fcn1/StatefulPartitionedCall_1+^encoder2/fcn2/mean/StatefulPartitionedCall-^encoder2/fcn2/mean/StatefulPartitionedCall_1&^encoder2/fcn3/StatefulPartitionedCall*
T0*#
_output_shapes
:         2

IdentityЦ

Identity_1Identity!tf.__operators__.add_18/AddV2:z:0&^decoder1/fcn1/StatefulPartitionedCall(^decoder1/fcn1/StatefulPartitionedCall_1&^decoder1/fcn2/StatefulPartitionedCall(^decoder1/fcn2/StatefulPartitionedCall_1,^decoder1/layer_gene/StatefulPartitionedCall.^decoder1/layer_gene/StatefulPartitionedCall_1&^decoder2/fcn1/StatefulPartitionedCall(^decoder2/fcn1/StatefulPartitionedCall_1&^decoder2/fcn2/StatefulPartitionedCall(^decoder2/fcn2/StatefulPartitionedCall_1&^encoder1/fcn1/StatefulPartitionedCall(^encoder1/fcn1/StatefulPartitionedCall_1+^encoder1/fcn2/mean/StatefulPartitionedCall-^encoder1/fcn2/mean/StatefulPartitionedCall_1&^encoder1/fcn3/StatefulPartitionedCall+^encoder1/fcn3_gene/StatefulPartitionedCall-^encoder1/fcn3_gene/StatefulPartitionedCall_1&^encoder2/fcn1/StatefulPartitionedCall(^encoder2/fcn1/StatefulPartitionedCall_1+^encoder2/fcn2/mean/StatefulPartitionedCall-^encoder2/fcn2/mean/StatefulPartitionedCall_1&^encoder2/fcn3/StatefulPartitionedCall*
T0*
_output_shapes
: 2

Identity_1┤

Identity_2Identity.encoder1/fcn3/StatefulPartitionedCall:output:0&^decoder1/fcn1/StatefulPartitionedCall(^decoder1/fcn1/StatefulPartitionedCall_1&^decoder1/fcn2/StatefulPartitionedCall(^decoder1/fcn2/StatefulPartitionedCall_1,^decoder1/layer_gene/StatefulPartitionedCall.^decoder1/layer_gene/StatefulPartitionedCall_1&^decoder2/fcn1/StatefulPartitionedCall(^decoder2/fcn1/StatefulPartitionedCall_1&^decoder2/fcn2/StatefulPartitionedCall(^decoder2/fcn2/StatefulPartitionedCall_1&^encoder1/fcn1/StatefulPartitionedCall(^encoder1/fcn1/StatefulPartitionedCall_1+^encoder1/fcn2/mean/StatefulPartitionedCall-^encoder1/fcn2/mean/StatefulPartitionedCall_1&^encoder1/fcn3/StatefulPartitionedCall+^encoder1/fcn3_gene/StatefulPartitionedCall-^encoder1/fcn3_gene/StatefulPartitionedCall_1&^encoder2/fcn1/StatefulPartitionedCall(^encoder2/fcn1/StatefulPartitionedCall_1+^encoder2/fcn2/mean/StatefulPartitionedCall-^encoder2/fcn2/mean/StatefulPartitionedCall_1&^encoder2/fcn3/StatefulPartitionedCall*
T0*'
_output_shapes
:         P2

Identity_2┤

Identity_3Identity.encoder2/fcn3/StatefulPartitionedCall:output:0&^decoder1/fcn1/StatefulPartitionedCall(^decoder1/fcn1/StatefulPartitionedCall_1&^decoder1/fcn2/StatefulPartitionedCall(^decoder1/fcn2/StatefulPartitionedCall_1,^decoder1/layer_gene/StatefulPartitionedCall.^decoder1/layer_gene/StatefulPartitionedCall_1&^decoder2/fcn1/StatefulPartitionedCall(^decoder2/fcn1/StatefulPartitionedCall_1&^decoder2/fcn2/StatefulPartitionedCall(^decoder2/fcn2/StatefulPartitionedCall_1&^encoder1/fcn1/StatefulPartitionedCall(^encoder1/fcn1/StatefulPartitionedCall_1+^encoder1/fcn2/mean/StatefulPartitionedCall-^encoder1/fcn2/mean/StatefulPartitionedCall_1&^encoder1/fcn3/StatefulPartitionedCall+^encoder1/fcn3_gene/StatefulPartitionedCall-^encoder1/fcn3_gene/StatefulPartitionedCall_1&^encoder2/fcn1/StatefulPartitionedCall(^encoder2/fcn1/StatefulPartitionedCall_1+^encoder2/fcn2/mean/StatefulPartitionedCall-^encoder2/fcn2/mean/StatefulPartitionedCall_1&^encoder2/fcn3/StatefulPartitionedCall*
T0*'
_output_shapes
:         P2

Identity_3"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0*(
_construction_contextkEagerRuntime*┐
_input_shapesн
к:         А:         А:         А:         А:         А:         А: : : : : : : : : : : : : : : : : : : : : : : : : 2N
%decoder1/fcn1/StatefulPartitionedCall%decoder1/fcn1/StatefulPartitionedCall2R
'decoder1/fcn1/StatefulPartitionedCall_1'decoder1/fcn1/StatefulPartitionedCall_12N
%decoder1/fcn2/StatefulPartitionedCall%decoder1/fcn2/StatefulPartitionedCall2R
'decoder1/fcn2/StatefulPartitionedCall_1'decoder1/fcn2/StatefulPartitionedCall_12Z
+decoder1/layer_gene/StatefulPartitionedCall+decoder1/layer_gene/StatefulPartitionedCall2^
-decoder1/layer_gene/StatefulPartitionedCall_1-decoder1/layer_gene/StatefulPartitionedCall_12N
%decoder2/fcn1/StatefulPartitionedCall%decoder2/fcn1/StatefulPartitionedCall2R
'decoder2/fcn1/StatefulPartitionedCall_1'decoder2/fcn1/StatefulPartitionedCall_12N
%decoder2/fcn2/StatefulPartitionedCall%decoder2/fcn2/StatefulPartitionedCall2R
'decoder2/fcn2/StatefulPartitionedCall_1'decoder2/fcn2/StatefulPartitionedCall_12N
%encoder1/fcn1/StatefulPartitionedCall%encoder1/fcn1/StatefulPartitionedCall2R
'encoder1/fcn1/StatefulPartitionedCall_1'encoder1/fcn1/StatefulPartitionedCall_12X
*encoder1/fcn2/mean/StatefulPartitionedCall*encoder1/fcn2/mean/StatefulPartitionedCall2\
,encoder1/fcn2/mean/StatefulPartitionedCall_1,encoder1/fcn2/mean/StatefulPartitionedCall_12N
%encoder1/fcn3/StatefulPartitionedCall%encoder1/fcn3/StatefulPartitionedCall2X
*encoder1/fcn3_gene/StatefulPartitionedCall*encoder1/fcn3_gene/StatefulPartitionedCall2\
,encoder1/fcn3_gene/StatefulPartitionedCall_1,encoder1/fcn3_gene/StatefulPartitionedCall_12N
%encoder2/fcn1/StatefulPartitionedCall%encoder2/fcn1/StatefulPartitionedCall2R
'encoder2/fcn1/StatefulPartitionedCall_1'encoder2/fcn1/StatefulPartitionedCall_12X
*encoder2/fcn2/mean/StatefulPartitionedCall*encoder2/fcn2/mean/StatefulPartitionedCall2\
,encoder2/fcn2/mean/StatefulPartitionedCall_1,encoder2/fcn2/mean/StatefulPartitionedCall_12N
%encoder2/fcn3/StatefulPartitionedCall%encoder2/fcn3/StatefulPartitionedCall:P L
(
_output_shapes
:         А
 
_user_specified_nameinputs:PL
(
_output_shapes
:         А
 
_user_specified_nameinputs:PL
(
_output_shapes
:         А
 
_user_specified_nameinputs:PL
(
_output_shapes
:         А
 
_user_specified_nameinputs:PL
(
_output_shapes
:         А
 
_user_specified_nameinputs:PL
(
_output_shapes
:         А
 
_user_specified_nameinputs:

_output_shapes
: 
Ц
Й
(__inference_model_layer_call_fn_40051395
inputs_0
inputs_1
inputs_2
inputs_3
inputs_4
inputs_5
unknown:	Аx
	unknown_0:x
	unknown_1:x`
	unknown_2:`
	unknown_3:	Аd
	unknown_4:d
	unknown_5:	Аd
	unknown_6:d
	unknown_7:dd
	unknown_8:d
	unknown_9:dd

unknown_10:d

unknown_11:	`А	

unknown_12:	А	

unknown_13:	dА	

unknown_14:	А	

unknown_15:
А	А

unknown_16:	А

unknown_17:
А	А

unknown_18:	А

unknown_19:`P

unknown_20:P

unknown_21:dP

unknown_22:P

unknown_23
identity

identity_1

identity_2

identity_3ИвStatefulPartitionedCallО
StatefulPartitionedCallStatefulPartitionedCallinputs_0inputs_1inputs_2inputs_3inputs_4inputs_5unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23**
Tin#
!2*
Tout
2*
_collective_manager_ids
 *K
_output_shapes9
7:         : :         P:         P*:
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8В *L
fGRE
C__inference_model_layer_call_and_return_conditional_losses_400502262
StatefulPartitionedCallК
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*#
_output_shapes
:         2

IdentityБ

Identity_1Identity StatefulPartitionedCall:output:1^StatefulPartitionedCall*
T0*
_output_shapes
: 2

Identity_1Т

Identity_2Identity StatefulPartitionedCall:output:2^StatefulPartitionedCall*
T0*'
_output_shapes
:         P2

Identity_2Т

Identity_3Identity StatefulPartitionedCall:output:3^StatefulPartitionedCall*
T0*'
_output_shapes
:         P2

Identity_3"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0*(
_construction_contextkEagerRuntime*┐
_input_shapesн
к:         А:         А:         А:         А:         А:         А: : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:R N
(
_output_shapes
:         А
"
_user_specified_name
inputs/0:RN
(
_output_shapes
:         А
"
_user_specified_name
inputs/1:RN
(
_output_shapes
:         А
"
_user_specified_name
inputs/2:RN
(
_output_shapes
:         А
"
_user_specified_name
inputs/3:RN
(
_output_shapes
:         А
"
_user_specified_name
inputs/4:RN
(
_output_shapes
:         А
"
_user_specified_name
inputs/5:

_output_shapes
: 
нН
Ж
C__inference_model_layer_call_and_return_conditional_losses_40050544
input_1
input_2
input_3
input_4
input_5
input_6)
encoder2_fcn1_40050359:	Аx$
encoder2_fcn1_40050361:x-
encoder2_fcn2_mean_40050364:x`)
encoder2_fcn2_mean_40050366:`.
encoder1_fcn3_gene_40050369:	Аd)
encoder1_fcn3_gene_40050371:d)
encoder1_fcn1_40050374:	Аd$
encoder1_fcn1_40050376:d-
encoder1_fcn2_mean_40050399:dd)
encoder1_fcn2_mean_40050401:d.
decoder1_layer_gene_40050414:dd*
decoder1_layer_gene_40050416:d)
decoder2_fcn1_40050433:	`А	%
decoder2_fcn1_40050435:	А	)
decoder1_fcn1_40050438:	dА	%
decoder1_fcn1_40050440:	А	*
decoder2_fcn2_40050453:
А	А%
decoder2_fcn2_40050455:	А*
decoder1_fcn2_40050460:
А	А%
decoder1_fcn2_40050462:	А(
encoder2_fcn3_40050493:`P$
encoder2_fcn3_40050495:P(
encoder1_fcn3_40050498:dP$
encoder1_fcn3_40050500:P
unknown
identity

identity_1

identity_2

identity_3Ив%decoder1/fcn1/StatefulPartitionedCallв'decoder1/fcn1/StatefulPartitionedCall_1в%decoder1/fcn2/StatefulPartitionedCallв'decoder1/fcn2/StatefulPartitionedCall_1в+decoder1/layer_gene/StatefulPartitionedCallв-decoder1/layer_gene/StatefulPartitionedCall_1в%decoder2/fcn1/StatefulPartitionedCallв'decoder2/fcn1/StatefulPartitionedCall_1в%decoder2/fcn2/StatefulPartitionedCallв'decoder2/fcn2/StatefulPartitionedCall_1в%encoder1/fcn1/StatefulPartitionedCallв'encoder1/fcn1/StatefulPartitionedCall_1в*encoder1/fcn2/mean/StatefulPartitionedCallв,encoder1/fcn2/mean/StatefulPartitionedCall_1в%encoder1/fcn3/StatefulPartitionedCallв*encoder1/fcn3_gene/StatefulPartitionedCallв,encoder1/fcn3_gene/StatefulPartitionedCall_1в%encoder2/fcn1/StatefulPartitionedCallв'encoder2/fcn1/StatefulPartitionedCall_1в*encoder2/fcn2/mean/StatefulPartitionedCallв,encoder2/fcn2/mean/StatefulPartitionedCall_1в%encoder2/fcn3/StatefulPartitionedCall╖
%encoder2/fcn1/StatefulPartitionedCallStatefulPartitionedCallinput_2encoder2_fcn1_40050359encoder2_fcn1_40050361*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         x*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *T
fORM
K__inference_encoder2/fcn1_layer_call_and_return_conditional_losses_400494682'
%encoder2/fcn1/StatefulPartitionedCallў
*encoder2/fcn2/mean/StatefulPartitionedCallStatefulPartitionedCall.encoder2/fcn1/StatefulPartitionedCall:output:0encoder2_fcn2_mean_40050364encoder2_fcn2_mean_40050366*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         `*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *Y
fTRR
P__inference_encoder2/fcn2/mean_layer_call_and_return_conditional_losses_400494852,
*encoder2/fcn2/mean/StatefulPartitionedCall╨
*encoder1/fcn3_gene/StatefulPartitionedCallStatefulPartitionedCallinput_4encoder1_fcn3_gene_40050369encoder1_fcn3_gene_40050371*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         d*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *Y
fTRR
P__inference_encoder1/fcn3_gene_layer_call_and_return_conditional_losses_400495012,
*encoder1/fcn3_gene/StatefulPartitionedCall╖
%encoder1/fcn1/StatefulPartitionedCallStatefulPartitionedCallinput_3encoder1_fcn1_40050374encoder1_fcn1_40050376*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         d*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *T
fORM
K__inference_encoder1/fcn1_layer_call_and_return_conditional_losses_400495182'
%encoder1/fcn1/StatefulPartitionedCall╘
,encoder1/fcn3_gene/StatefulPartitionedCall_1StatefulPartitionedCallinput_6encoder1_fcn3_gene_40050369encoder1_fcn3_gene_40050371*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         d*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *Y
fTRR
P__inference_encoder1/fcn3_gene_layer_call_and_return_conditional_losses_400495012.
,encoder1/fcn3_gene/StatefulPartitionedCall_1╗
'encoder1/fcn1/StatefulPartitionedCall_1StatefulPartitionedCallinput_1encoder1_fcn1_40050374encoder1_fcn1_40050376*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         d*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *T
fORM
K__inference_encoder1/fcn1_layer_call_and_return_conditional_losses_400495182)
'encoder1/fcn1/StatefulPartitionedCall_1г
tf.math.square_5/SquareSquare3encoder2/fcn2/mean/StatefulPartitionedCall:output:0*
T0*'
_output_shapes
:         `2
tf.math.square_5/Squareг
tf.math.square_4/SquareSquare3encoder2/fcn2/mean/StatefulPartitionedCall:output:0*
T0*'
_output_shapes
:         `2
tf.math.square_4/Squarex
tf.math.square_3/SquareSquareinput_2*
T0*(
_output_shapes
:         А2
tf.math.square_3/Squarex
tf.math.square_2/SquareSquareinput_2*
T0*(
_output_shapes
:         А2
tf.math.square_2/Square╬
tf.math.subtract_3/SubSub.encoder1/fcn1/StatefulPartitionedCall:output:03encoder1/fcn3_gene/StatefulPartitionedCall:output:0*
T0*'
_output_shapes
:         d2
tf.math.subtract_3/Sub╬
tf.math.subtract/SubSub0encoder1/fcn1/StatefulPartitionedCall_1:output:05encoder1/fcn3_gene/StatefulPartitionedCall_1:output:0*
T0*'
_output_shapes
:         d2
tf.math.subtract/Subг
*tf.math.reduce_sum_3/Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
valueB :
         2,
*tf.math.reduce_sum_3/Sum/reduction_indices╨
tf.math.reduce_sum_3/SumSumtf.math.square_5/Square:y:03tf.math.reduce_sum_3/Sum/reduction_indices:output:0*
T0*'
_output_shapes
:         *
	keep_dims(2
tf.math.reduce_sum_3/Sumг
*tf.math.reduce_sum_2/Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
valueB :
         2,
*tf.math.reduce_sum_2/Sum/reduction_indices╨
tf.math.reduce_sum_2/SumSumtf.math.square_4/Square:y:03tf.math.reduce_sum_2/Sum/reduction_indices:output:0*
T0*'
_output_shapes
:         *
	keep_dims(2
tf.math.reduce_sum_2/Sumг
*tf.math.reduce_sum_1/Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
valueB :
         2,
*tf.math.reduce_sum_1/Sum/reduction_indices╨
tf.math.reduce_sum_1/SumSumtf.math.square_3/Square:y:03tf.math.reduce_sum_1/Sum/reduction_indices:output:0*
T0*'
_output_shapes
:         *
	keep_dims(2
tf.math.reduce_sum_1/SumЯ
(tf.math.reduce_sum/Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
valueB :
         2*
(tf.math.reduce_sum/Sum/reduction_indices╩
tf.math.reduce_sum/SumSumtf.math.square_2/Square:y:01tf.math.reduce_sum/Sum/reduction_indices:output:0*
T0*'
_output_shapes
:         *
	keep_dims(2
tf.math.reduce_sum/Sumу
*encoder1/fcn2/mean/StatefulPartitionedCallStatefulPartitionedCalltf.math.subtract_3/Sub:z:0encoder1_fcn2_mean_40050399encoder1_fcn2_mean_40050401*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         d*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *Y
fTRR
P__inference_encoder1/fcn2/mean_layer_call_and_return_conditional_losses_400495552,
*encoder1/fcn2/mean/StatefulPartitionedCallх
,encoder1/fcn2/mean/StatefulPartitionedCall_1StatefulPartitionedCalltf.math.subtract/Sub:z:0encoder1_fcn2_mean_40050399encoder1_fcn2_mean_40050401*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         d*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *Y
fTRR
P__inference_encoder1/fcn2/mean_layer_call_and_return_conditional_losses_400495552.
,encoder1/fcn2/mean/StatefulPartitionedCall_1З
tf.math.sqrt_2/SqrtSqrt!tf.math.reduce_sum_2/Sum:output:0*
T0*'
_output_shapes
:         2
tf.math.sqrt_2/SqrtЗ
tf.math.sqrt_3/SqrtSqrt!tf.math.reduce_sum_3/Sum:output:0*
T0*'
_output_shapes
:         2
tf.math.sqrt_3/SqrtБ
tf.math.sqrt/SqrtSqrttf.math.reduce_sum/Sum:output:0*
T0*'
_output_shapes
:         2
tf.math.sqrt/SqrtЗ
tf.math.sqrt_1/SqrtSqrt!tf.math.reduce_sum_1/Sum:output:0*
T0*'
_output_shapes
:         2
tf.math.sqrt_1/Sqrt╗
'encoder2/fcn1/StatefulPartitionedCall_1StatefulPartitionedCallinput_5encoder2_fcn1_40050359encoder2_fcn1_40050361*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         x*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *T
fORM
K__inference_encoder2/fcn1_layer_call_and_return_conditional_losses_400494682)
'encoder2/fcn1/StatefulPartitionedCall_1Б
+decoder1/layer_gene/StatefulPartitionedCallStatefulPartitionedCall3encoder1/fcn2/mean/StatefulPartitionedCall:output:0decoder1_layer_gene_40050414decoder1_layer_gene_40050416*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         d*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *Z
fURS
Q__inference_decoder1/layer_gene_layer_call_and_return_conditional_losses_400495822-
+decoder1/layer_gene/StatefulPartitionedCallЗ
-decoder1/layer_gene/StatefulPartitionedCall_1StatefulPartitionedCall5encoder1/fcn2/mean/StatefulPartitionedCall_1:output:0decoder1_layer_gene_40050414decoder1_layer_gene_40050416*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         d*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *Z
fURS
Q__inference_decoder1/layer_gene_layer_call_and_return_conditional_losses_400495822/
-decoder1/layer_gene/StatefulPartitionedCall_1°
tf.linalg.matmul_1/MatMulMatMul3encoder2/fcn2/mean/StatefulPartitionedCall:output:03encoder2/fcn2/mean/StatefulPartitionedCall:output:0*
T0*0
_output_shapes
:                  *
transpose_b(2
tf.linalg.matmul_1/MatMulЫ
tf.math.multiply_3/MulMultf.math.sqrt_2/Sqrt:y:0tf.math.sqrt_3/Sqrt:y:0*
T0*'
_output_shapes
:         2
tf.math.multiply_3/MulЬ
tf.linalg.matmul/MatMulMatMulinput_2input_2*
T0*0
_output_shapes
:                  *
transpose_b(2
tf.linalg.matmul/MatMulЩ
tf.math.multiply_2/MulMultf.math.sqrt/Sqrt:y:0tf.math.sqrt_1/Sqrt:y:0*
T0*'
_output_shapes
:         2
tf.math.multiply_2/Mul¤
,encoder2/fcn2/mean/StatefulPartitionedCall_1StatefulPartitionedCall0encoder2/fcn1/StatefulPartitionedCall_1:output:0encoder2_fcn2_mean_40050364encoder2_fcn2_mean_40050366*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         `*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *Y
fTRR
P__inference_encoder2/fcn2/mean_layer_call_and_return_conditional_losses_400494852.
,encoder2/fcn2/mean/StatefulPartitionedCall_1т
tf.__operators__.add_2/AddV2AddV24decoder1/layer_gene/StatefulPartitionedCall:output:03encoder1/fcn3_gene/StatefulPartitionedCall:output:0*
T0*'
_output_shapes
:         d2
tf.__operators__.add_2/AddV2т
tf.__operators__.add/AddV2AddV26decoder1/layer_gene/StatefulPartitionedCall_1:output:05encoder1/fcn3_gene/StatefulPartitionedCall_1:output:0*
T0*'
_output_shapes
:         d2
tf.__operators__.add/AddV2╜
tf.math.truediv_1/truedivRealDiv#tf.linalg.matmul_1/MatMul:product:0tf.math.multiply_3/Mul:z:0*
T0*0
_output_shapes
:                  2
tf.math.truediv_1/truediv╖
tf.math.truediv/truedivRealDiv!tf.linalg.matmul/MatMul:product:0tf.math.multiply_2/Mul:z:0*
T0*0
_output_shapes
:                  2
tf.math.truediv/truedivц
%decoder2/fcn1/StatefulPartitionedCallStatefulPartitionedCall5encoder2/fcn2/mean/StatefulPartitionedCall_1:output:0decoder2_fcn1_40050433decoder2_fcn1_40050435*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         А	*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *T
fORM
K__inference_decoder2/fcn1_layer_call_and_return_conditional_losses_400496132'
%decoder2/fcn1/StatefulPartitionedCall╤
%decoder1/fcn1/StatefulPartitionedCallStatefulPartitionedCall tf.__operators__.add_2/AddV2:z:0decoder1_fcn1_40050438decoder1_fcn1_40050440*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         А	*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *T
fORM
K__inference_decoder1/fcn1_layer_call_and_return_conditional_losses_400496302'
%decoder1/fcn1/StatefulPartitionedCallш
'decoder2/fcn1/StatefulPartitionedCall_1StatefulPartitionedCall3encoder2/fcn2/mean/StatefulPartitionedCall:output:0decoder2_fcn1_40050433decoder2_fcn1_40050435*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         А	*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *T
fORM
K__inference_decoder2/fcn1_layer_call_and_return_conditional_losses_400496132)
'decoder2/fcn1/StatefulPartitionedCall_1╙
'decoder1/fcn1/StatefulPartitionedCall_1StatefulPartitionedCalltf.__operators__.add/AddV2:z:0decoder1_fcn1_40050438decoder1_fcn1_40050440*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         А	*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *T
fORM
K__inference_decoder1/fcn1_layer_call_and_return_conditional_losses_400496302)
'decoder1/fcn1/StatefulPartitionedCall_1Ч
tf.nn.softmax_1/SoftmaxSoftmaxtf.math.truediv_1/truediv:z:0*
T0*0
_output_shapes
:                  2
tf.nn.softmax_1/SoftmaxС
tf.nn.softmax/SoftmaxSoftmaxtf.math.truediv/truediv:z:0*
T0*0
_output_shapes
:                  2
tf.nn.softmax/Softmaxy
tf.math.multiply_9/Mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *  HB2
tf.math.multiply_9/Mul/yЦ
tf.math.multiply_9/MulMulinput_5!tf.math.multiply_9/Mul/y:output:0*
T0*(
_output_shapes
:         А2
tf.math.multiply_9/Mul▀
%decoder2/fcn2/StatefulPartitionedCallStatefulPartitionedCall.decoder2/fcn1/StatefulPartitionedCall:output:0decoder2_fcn2_40050453decoder2_fcn2_40050455*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         А*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *T
fORM
K__inference_decoder2/fcn2_layer_call_and_return_conditional_losses_400496562'
%decoder2/fcn2/StatefulPartitionedCally
tf.math.multiply_8/Mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *  HB2
tf.math.multiply_8/Mul/yЦ
tf.math.multiply_8/MulMulinput_3!tf.math.multiply_8/Mul/y:output:0*
T0*(
_output_shapes
:         А2
tf.math.multiply_8/Mul▀
%decoder1/fcn2/StatefulPartitionedCallStatefulPartitionedCall.decoder1/fcn1/StatefulPartitionedCall:output:0decoder1_fcn2_40050460decoder1_fcn2_40050462*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         А*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *T
fORM
K__inference_decoder1/fcn2_layer_call_and_return_conditional_losses_400496742'
%decoder1/fcn2/StatefulPartitionedCallх
'decoder2/fcn2/StatefulPartitionedCall_1StatefulPartitionedCall0decoder2/fcn1/StatefulPartitionedCall_1:output:0decoder2_fcn2_40050453decoder2_fcn2_40050455*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         А*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *T
fORM
K__inference_decoder2/fcn2_layer_call_and_return_conditional_losses_400496562)
'decoder2/fcn2/StatefulPartitionedCall_1х
'decoder1/fcn2/StatefulPartitionedCall_1StatefulPartitionedCall0decoder1/fcn1/StatefulPartitionedCall_1:output:0decoder1_fcn2_40050460decoder1_fcn2_40050462*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         А*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *T
fORM
K__inference_decoder1/fcn2_layer_call_and_return_conditional_losses_400496742)
'decoder1/fcn2/StatefulPartitionedCall_1└
tf.math.truediv_3/truedivRealDiv!tf.nn.softmax_1/Softmax:softmax:0tf.nn.softmax/Softmax:softmax:0*
T0*0
_output_shapes
:                  2
tf.math.truediv_3/truediv└
tf.math.truediv_2/truedivRealDivtf.nn.softmax/Softmax:softmax:0!tf.nn.softmax_1/Softmax:softmax:0*
T0*0
_output_shapes
:                  2
tf.math.truediv_2/truediv╢
tf.math.subtract_5/SubSubtf.math.multiply_9/Mul:z:0.decoder2/fcn2/StatefulPartitionedCall:output:0*
T0*(
_output_shapes
:         А2
tf.math.subtract_5/Sub╢
tf.math.subtract_4/SubSubtf.math.multiply_8/Mul:z:0.decoder1/fcn2/StatefulPartitionedCall:output:0*
T0*(
_output_shapes
:         А2
tf.math.subtract_4/Suby
tf.math.multiply_1/Mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *  HB2
tf.math.multiply_1/Mul/yЦ
tf.math.multiply_1/MulMulinput_2!tf.math.multiply_1/Mul/y:output:0*
T0*(
_output_shapes
:         А2
tf.math.multiply_1/Mulu
tf.math.multiply/Mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *  HB2
tf.math.multiply/Mul/yР
tf.math.multiply/MulMulinput_1tf.math.multiply/Mul/y:output:0*
T0*(
_output_shapes
:         А2
tf.math.multiply/MulЗ
tf.math.log_1/LogLogtf.math.truediv_3/truediv:z:0*
T0*0
_output_shapes
:                  2
tf.math.log_1/LogГ
tf.math.log/LogLogtf.math.truediv_2/truediv:z:0*
T0*0
_output_shapes
:                  2
tf.math.log/LogЛ
tf.math.square_9/SquareSquaretf.math.subtract_5/Sub:z:0*
T0*(
_output_shapes
:         А2
tf.math.square_9/SquareЛ
tf.math.square_8/SquareSquaretf.math.subtract_4/Sub:z:0*
T0*(
_output_shapes
:         А2
tf.math.square_8/Square╕
tf.math.subtract_2/SubSubtf.math.multiply_1/Mul:z:00decoder2/fcn2/StatefulPartitionedCall_1:output:0*
T0*(
_output_shapes
:         А2
tf.math.subtract_2/Sub╢
tf.math.subtract_1/SubSubtf.math.multiply/Mul:z:00decoder1/fcn2/StatefulPartitionedCall_1:output:0*
T0*(
_output_shapes
:         А2
tf.math.subtract_1/Subм
tf.math.multiply_5/MulMul!tf.nn.softmax_1/Softmax:softmax:0tf.math.log_1/Log:y:0*
T0*0
_output_shapes
:                  2
tf.math.multiply_5/Mulи
tf.math.multiply_4/MulMultf.nn.softmax/Softmax:softmax:0tf.math.log/Log:y:0*
T0*0
_output_shapes
:                  2
tf.math.multiply_4/MulЛ
tf.math.reduce_mean_3/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2
tf.math.reduce_mean_3/Constд
tf.math.reduce_mean_3/MeanMeantf.math.square_8/Square:y:0$tf.math.reduce_mean_3/Const:output:0*
T0*
_output_shapes
: 2
tf.math.reduce_mean_3/MeanЛ
tf.math.reduce_mean_4/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2
tf.math.reduce_mean_4/Constд
tf.math.reduce_mean_4/MeanMeantf.math.square_9/Square:y:0$tf.math.reduce_mean_4/Const:output:0*
T0*
_output_shapes
: 2
tf.math.reduce_mean_4/MeanЛ
tf.math.square_1/SquareSquaretf.math.subtract_2/Sub:z:0*
T0*(
_output_shapes
:         А2
tf.math.square_1/SquareЗ
tf.math.square/SquareSquaretf.math.subtract_1/Sub:z:0*
T0*(
_output_shapes
:         А2
tf.math.square/Squareу
%encoder2/fcn3/StatefulPartitionedCallStatefulPartitionedCall3encoder2/fcn2/mean/StatefulPartitionedCall:output:0encoder2_fcn3_40050493encoder2_fcn3_40050495*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         P*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *T
fORM
K__inference_encoder2/fcn3_layer_call_and_return_conditional_losses_400497182'
%encoder2/fcn3/StatefulPartitionedCallх
%encoder1/fcn3/StatefulPartitionedCallStatefulPartitionedCall5encoder1/fcn2/mean/StatefulPartitionedCall_1:output:0encoder1_fcn3_40050498encoder1_fcn3_40050500*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         P*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *T
fORM
K__inference_encoder1/fcn3_layer_call_and_return_conditional_losses_400497342'
%encoder1/fcn3/StatefulPartitionedCallг
*tf.math.reduce_sum_4/Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
valueB :
         2,
*tf.math.reduce_sum_4/Sum/reduction_indices║
tf.math.reduce_sum_4/SumSumtf.math.multiply_4/Mul:z:03tf.math.reduce_sum_4/Sum/reduction_indices:output:0*
T0*#
_output_shapes
:         2
tf.math.reduce_sum_4/Sumг
*tf.math.reduce_sum_5/Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
valueB :
         2,
*tf.math.reduce_sum_5/Sum/reduction_indices║
tf.math.reduce_sum_5/SumSumtf.math.multiply_5/Mul:z:03tf.math.reduce_sum_5/Sum/reduction_indices:output:0*
T0*#
_output_shapes
:         2
tf.math.reduce_sum_5/Sum▓
tf.__operators__.add_16/AddV2AddV2#tf.math.reduce_mean_3/Mean:output:0#tf.math.reduce_mean_4/Mean:output:0*
T0*
_output_shapes
: 2
tf.__operators__.add_16/AddV2З
tf.math.reduce_mean/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2
tf.math.reduce_mean/ConstЬ
tf.math.reduce_mean/MeanMeantf.math.square/Square:y:0"tf.math.reduce_mean/Const:output:0*
T0*
_output_shapes
: 2
tf.math.reduce_mean/MeanЛ
tf.math.reduce_mean_1/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2
tf.math.reduce_mean_1/Constд
tf.math.reduce_mean_1/MeanMeantf.math.square_1/Square:y:0$tf.math.reduce_mean_1/Const:output:0*
T0*
_output_shapes
: 2
tf.math.reduce_mean_1/MeanЮ
tf.math.square_7/SquareSquare.encoder2/fcn3/StatefulPartitionedCall:output:0*
T0*'
_output_shapes
:         P2
tf.math.square_7/SquareЮ
tf.math.square_6/SquareSquare.encoder1/fcn3/StatefulPartitionedCall:output:0*
T0*'
_output_shapes
:         P2
tf.math.square_6/Square╣
tf.__operators__.add_1/AddV2AddV2!tf.math.reduce_sum_4/Sum:output:0!tf.math.reduce_sum_5/Sum:output:0*
T0*#
_output_shapes
:         2
tf.__operators__.add_1/AddV2Ж
tf.math.multiply_22/MulMulunknown!tf.__operators__.add_16/AddV2:z:0*
T0*
_output_shapes
: 2
tf.math.multiply_22/Mul░
tf.__operators__.add_15/AddV2AddV2!tf.math.reduce_mean/Mean:output:0#tf.math.reduce_mean_1/Mean:output:0*
T0*
_output_shapes
: 2
tf.__operators__.add_15/AddV2г
*tf.math.reduce_sum_8/Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
valueB :
         2,
*tf.math.reduce_sum_8/Sum/reduction_indices╗
tf.math.reduce_sum_8/SumSumtf.math.square_7/Square:y:03tf.math.reduce_sum_8/Sum/reduction_indices:output:0*
T0*#
_output_shapes
:         2
tf.math.reduce_sum_8/Sumг
*tf.math.reduce_sum_7/Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
valueB :
         2,
*tf.math.reduce_sum_7/Sum/reduction_indices╗
tf.math.reduce_sum_7/SumSumtf.math.square_6/Square:y:03tf.math.reduce_sum_7/Sum/reduction_indices:output:0*
T0*#
_output_shapes
:         2
tf.math.reduce_sum_7/SumД
tf.math.reduce_mean_2/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2
tf.math.reduce_mean_2/Constй
tf.math.reduce_mean_2/MeanMean tf.__operators__.add_1/AddV2:z:0$tf.math.reduce_mean_2/Const:output:0*
T0*
_output_shapes
: 2
tf.math.reduce_mean_2/MeanБ
tf.math.truediv_21/truediv/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
tf.math.truediv_21/truediv/yо
tf.math.truediv_21/truedivRealDiv!tf.__operators__.add_15/AddV2:z:0%tf.math.truediv_21/truediv/y:output:0*
T0*
_output_shapes
: 2
tf.math.truediv_21/truedivБ
tf.math.truediv_22/truediv/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
tf.math.truediv_22/truediv/yи
tf.math.truediv_22/truedivRealDivtf.math.multiply_22/Mul:z:0%tf.math.truediv_22/truediv/y:output:0*
T0*
_output_shapes
: 2
tf.math.truediv_22/truedivГ
tf.math.sqrt_4/SqrtSqrt!tf.math.reduce_sum_7/Sum:output:0*
T0*#
_output_shapes
:         2
tf.math.sqrt_4/SqrtГ
tf.math.sqrt_5/SqrtSqrt!tf.math.reduce_sum_8/Sum:output:0*
T0*#
_output_shapes
:         2
tf.math.sqrt_5/Sqrt╔
tf.math.multiply_6/MulMul.encoder1/fcn3/StatefulPartitionedCall:output:0.encoder2/fcn3/StatefulPartitionedCall:output:0*
T0*'
_output_shapes
:         P2
tf.math.multiply_6/Mulи
tf.__operators__.add_17/AddV2AddV2tf.math.truediv_21/truediv:z:0tf.math.truediv_22/truediv:z:0*
T0*
_output_shapes
: 2
tf.__operators__.add_17/AddV2{
tf.math.multiply_23/Mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2
tf.math.multiply_23/Mul/yг
tf.math.multiply_23/MulMul#tf.math.reduce_mean_2/Mean:output:0"tf.math.multiply_23/Mul/y:output:0*
T0*
_output_shapes
: 2
tf.math.multiply_23/Mulг
*tf.math.reduce_sum_6/Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
valueB :
         2,
*tf.math.reduce_sum_6/Sum/reduction_indices║
tf.math.reduce_sum_6/SumSumtf.math.multiply_6/Mul:z:03tf.math.reduce_sum_6/Sum/reduction_indices:output:0*
T0*#
_output_shapes
:         2
tf.math.reduce_sum_6/SumЧ
tf.math.multiply_7/MulMultf.math.sqrt_4/Sqrt:y:0tf.math.sqrt_5/Sqrt:y:0*
T0*#
_output_shapes
:         2
tf.math.multiply_7/Mulи
tf.__operators__.add_18/AddV2AddV2!tf.__operators__.add_17/AddV2:z:0tf.math.multiply_23/Mul:z:0*
T0*
_output_shapes
: 2
tf.__operators__.add_18/AddV2о
tf.math.truediv_4/truedivRealDiv!tf.math.reduce_sum_6/Sum:output:0tf.math.multiply_7/Mul:z:0*
T0*#
_output_shapes
:         2
tf.math.truediv_4/truedivЫ
IdentityIdentitytf.math.truediv_4/truediv:z:0&^decoder1/fcn1/StatefulPartitionedCall(^decoder1/fcn1/StatefulPartitionedCall_1&^decoder1/fcn2/StatefulPartitionedCall(^decoder1/fcn2/StatefulPartitionedCall_1,^decoder1/layer_gene/StatefulPartitionedCall.^decoder1/layer_gene/StatefulPartitionedCall_1&^decoder2/fcn1/StatefulPartitionedCall(^decoder2/fcn1/StatefulPartitionedCall_1&^decoder2/fcn2/StatefulPartitionedCall(^decoder2/fcn2/StatefulPartitionedCall_1&^encoder1/fcn1/StatefulPartitionedCall(^encoder1/fcn1/StatefulPartitionedCall_1+^encoder1/fcn2/mean/StatefulPartitionedCall-^encoder1/fcn2/mean/StatefulPartitionedCall_1&^encoder1/fcn3/StatefulPartitionedCall+^encoder1/fcn3_gene/StatefulPartitionedCall-^encoder1/fcn3_gene/StatefulPartitionedCall_1&^encoder2/fcn1/StatefulPartitionedCall(^encoder2/fcn1/StatefulPartitionedCall_1+^encoder2/fcn2/mean/StatefulPartitionedCall-^encoder2/fcn2/mean/StatefulPartitionedCall_1&^encoder2/fcn3/StatefulPartitionedCall*
T0*#
_output_shapes
:         2

IdentityЦ

Identity_1Identity!tf.__operators__.add_18/AddV2:z:0&^decoder1/fcn1/StatefulPartitionedCall(^decoder1/fcn1/StatefulPartitionedCall_1&^decoder1/fcn2/StatefulPartitionedCall(^decoder1/fcn2/StatefulPartitionedCall_1,^decoder1/layer_gene/StatefulPartitionedCall.^decoder1/layer_gene/StatefulPartitionedCall_1&^decoder2/fcn1/StatefulPartitionedCall(^decoder2/fcn1/StatefulPartitionedCall_1&^decoder2/fcn2/StatefulPartitionedCall(^decoder2/fcn2/StatefulPartitionedCall_1&^encoder1/fcn1/StatefulPartitionedCall(^encoder1/fcn1/StatefulPartitionedCall_1+^encoder1/fcn2/mean/StatefulPartitionedCall-^encoder1/fcn2/mean/StatefulPartitionedCall_1&^encoder1/fcn3/StatefulPartitionedCall+^encoder1/fcn3_gene/StatefulPartitionedCall-^encoder1/fcn3_gene/StatefulPartitionedCall_1&^encoder2/fcn1/StatefulPartitionedCall(^encoder2/fcn1/StatefulPartitionedCall_1+^encoder2/fcn2/mean/StatefulPartitionedCall-^encoder2/fcn2/mean/StatefulPartitionedCall_1&^encoder2/fcn3/StatefulPartitionedCall*
T0*
_output_shapes
: 2

Identity_1┤

Identity_2Identity.encoder1/fcn3/StatefulPartitionedCall:output:0&^decoder1/fcn1/StatefulPartitionedCall(^decoder1/fcn1/StatefulPartitionedCall_1&^decoder1/fcn2/StatefulPartitionedCall(^decoder1/fcn2/StatefulPartitionedCall_1,^decoder1/layer_gene/StatefulPartitionedCall.^decoder1/layer_gene/StatefulPartitionedCall_1&^decoder2/fcn1/StatefulPartitionedCall(^decoder2/fcn1/StatefulPartitionedCall_1&^decoder2/fcn2/StatefulPartitionedCall(^decoder2/fcn2/StatefulPartitionedCall_1&^encoder1/fcn1/StatefulPartitionedCall(^encoder1/fcn1/StatefulPartitionedCall_1+^encoder1/fcn2/mean/StatefulPartitionedCall-^encoder1/fcn2/mean/StatefulPartitionedCall_1&^encoder1/fcn3/StatefulPartitionedCall+^encoder1/fcn3_gene/StatefulPartitionedCall-^encoder1/fcn3_gene/StatefulPartitionedCall_1&^encoder2/fcn1/StatefulPartitionedCall(^encoder2/fcn1/StatefulPartitionedCall_1+^encoder2/fcn2/mean/StatefulPartitionedCall-^encoder2/fcn2/mean/StatefulPartitionedCall_1&^encoder2/fcn3/StatefulPartitionedCall*
T0*'
_output_shapes
:         P2

Identity_2┤

Identity_3Identity.encoder2/fcn3/StatefulPartitionedCall:output:0&^decoder1/fcn1/StatefulPartitionedCall(^decoder1/fcn1/StatefulPartitionedCall_1&^decoder1/fcn2/StatefulPartitionedCall(^decoder1/fcn2/StatefulPartitionedCall_1,^decoder1/layer_gene/StatefulPartitionedCall.^decoder1/layer_gene/StatefulPartitionedCall_1&^decoder2/fcn1/StatefulPartitionedCall(^decoder2/fcn1/StatefulPartitionedCall_1&^decoder2/fcn2/StatefulPartitionedCall(^decoder2/fcn2/StatefulPartitionedCall_1&^encoder1/fcn1/StatefulPartitionedCall(^encoder1/fcn1/StatefulPartitionedCall_1+^encoder1/fcn2/mean/StatefulPartitionedCall-^encoder1/fcn2/mean/StatefulPartitionedCall_1&^encoder1/fcn3/StatefulPartitionedCall+^encoder1/fcn3_gene/StatefulPartitionedCall-^encoder1/fcn3_gene/StatefulPartitionedCall_1&^encoder2/fcn1/StatefulPartitionedCall(^encoder2/fcn1/StatefulPartitionedCall_1+^encoder2/fcn2/mean/StatefulPartitionedCall-^encoder2/fcn2/mean/StatefulPartitionedCall_1&^encoder2/fcn3/StatefulPartitionedCall*
T0*'
_output_shapes
:         P2

Identity_3"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0*(
_construction_contextkEagerRuntime*┐
_input_shapesн
к:         А:         А:         А:         А:         А:         А: : : : : : : : : : : : : : : : : : : : : : : : : 2N
%decoder1/fcn1/StatefulPartitionedCall%decoder1/fcn1/StatefulPartitionedCall2R
'decoder1/fcn1/StatefulPartitionedCall_1'decoder1/fcn1/StatefulPartitionedCall_12N
%decoder1/fcn2/StatefulPartitionedCall%decoder1/fcn2/StatefulPartitionedCall2R
'decoder1/fcn2/StatefulPartitionedCall_1'decoder1/fcn2/StatefulPartitionedCall_12Z
+decoder1/layer_gene/StatefulPartitionedCall+decoder1/layer_gene/StatefulPartitionedCall2^
-decoder1/layer_gene/StatefulPartitionedCall_1-decoder1/layer_gene/StatefulPartitionedCall_12N
%decoder2/fcn1/StatefulPartitionedCall%decoder2/fcn1/StatefulPartitionedCall2R
'decoder2/fcn1/StatefulPartitionedCall_1'decoder2/fcn1/StatefulPartitionedCall_12N
%decoder2/fcn2/StatefulPartitionedCall%decoder2/fcn2/StatefulPartitionedCall2R
'decoder2/fcn2/StatefulPartitionedCall_1'decoder2/fcn2/StatefulPartitionedCall_12N
%encoder1/fcn1/StatefulPartitionedCall%encoder1/fcn1/StatefulPartitionedCall2R
'encoder1/fcn1/StatefulPartitionedCall_1'encoder1/fcn1/StatefulPartitionedCall_12X
*encoder1/fcn2/mean/StatefulPartitionedCall*encoder1/fcn2/mean/StatefulPartitionedCall2\
,encoder1/fcn2/mean/StatefulPartitionedCall_1,encoder1/fcn2/mean/StatefulPartitionedCall_12N
%encoder1/fcn3/StatefulPartitionedCall%encoder1/fcn3/StatefulPartitionedCall2X
*encoder1/fcn3_gene/StatefulPartitionedCall*encoder1/fcn3_gene/StatefulPartitionedCall2\
,encoder1/fcn3_gene/StatefulPartitionedCall_1,encoder1/fcn3_gene/StatefulPartitionedCall_12N
%encoder2/fcn1/StatefulPartitionedCall%encoder2/fcn1/StatefulPartitionedCall2R
'encoder2/fcn1/StatefulPartitionedCall_1'encoder2/fcn1/StatefulPartitionedCall_12X
*encoder2/fcn2/mean/StatefulPartitionedCall*encoder2/fcn2/mean/StatefulPartitionedCall2\
,encoder2/fcn2/mean/StatefulPartitionedCall_1,encoder2/fcn2/mean/StatefulPartitionedCall_12N
%encoder2/fcn3/StatefulPartitionedCall%encoder2/fcn3/StatefulPartitionedCall:Q M
(
_output_shapes
:         А
!
_user_specified_name	input_1:QM
(
_output_shapes
:         А
!
_user_specified_name	input_2:QM
(
_output_shapes
:         А
!
_user_specified_name	input_3:QM
(
_output_shapes
:         А
!
_user_specified_name	input_4:QM
(
_output_shapes
:         А
!
_user_specified_name	input_5:QM
(
_output_shapes
:         А
!
_user_specified_name	input_6:

_output_shapes
: 
╖

¤
K__inference_encoder1/fcn1_layer_call_and_return_conditional_losses_40049518

inputs1
matmul_readvariableop_resource:	Аd-
biasadd_readvariableop_resource:d
identityИвBiasAdd/ReadVariableOpвMatMul/ReadVariableOpО
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	Аd*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         d2
MatMulМ
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:d*
dtype02
BiasAdd/ReadVariableOpБ
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         d2	
BiasAddX
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:         d2
ReluЧ
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:         d2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:         А: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:         А
 
_user_specified_nameinputs
╕
г
6__inference_decoder1/layer_gene_layer_call_fn_40051514

inputs
unknown:dd
	unknown_0:d
identityИвStatefulPartitionedCallД
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         d*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *Z
fURS
Q__inference_decoder1/layer_gene_layer_call_and_return_conditional_losses_400495822
StatefulPartitionedCallО
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:         d2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:         d: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:         d
 
_user_specified_nameinputs
│
а
0__inference_decoder1/fcn2_layer_call_fn_40051573

inputs
unknown:
А	А
	unknown_0:	А
identityИвStatefulPartitionedCall 
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         А*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *T
fORM
K__inference_decoder1/fcn2_layer_call_and_return_conditional_losses_400496742
StatefulPartitionedCallП
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:         А2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:         А	: : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:         А	
 
_user_specified_nameinputs
р	
В
P__inference_encoder1/fcn3_gene_layer_call_and_return_conditional_losses_40049501

inputs1
matmul_readvariableop_resource:	Аd-
biasadd_readvariableop_resource:d
identityИвBiasAdd/ReadVariableOpвMatMul/ReadVariableOpО
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	Аd*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         d2
MatMulМ
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:d*
dtype02
BiasAdd/ReadVariableOpБ
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         d2	
BiasAddХ
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:         d2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:         А: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:         А
 
_user_specified_nameinputs
Д
Г
(__inference_model_layer_call_fn_40049840
input_1
input_2
input_3
input_4
input_5
input_6
unknown:	Аx
	unknown_0:x
	unknown_1:x`
	unknown_2:`
	unknown_3:	Аd
	unknown_4:d
	unknown_5:	Аd
	unknown_6:d
	unknown_7:dd
	unknown_8:d
	unknown_9:dd

unknown_10:d

unknown_11:	`А	

unknown_12:	А	

unknown_13:	dА	

unknown_14:	А	

unknown_15:
А	А

unknown_16:	А

unknown_17:
А	А

unknown_18:	А

unknown_19:`P

unknown_20:P

unknown_21:dP

unknown_22:P

unknown_23
identity

identity_1

identity_2

identity_3ИвStatefulPartitionedCallИ
StatefulPartitionedCallStatefulPartitionedCallinput_1input_2input_3input_4input_5input_6unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23**
Tin#
!2*
Tout
2*
_collective_manager_ids
 *K
_output_shapes9
7:         : :         P:         P*:
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8В *L
fGRE
C__inference_model_layer_call_and_return_conditional_losses_400497812
StatefulPartitionedCallК
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*#
_output_shapes
:         2

IdentityБ

Identity_1Identity StatefulPartitionedCall:output:1^StatefulPartitionedCall*
T0*
_output_shapes
: 2

Identity_1Т

Identity_2Identity StatefulPartitionedCall:output:2^StatefulPartitionedCall*
T0*'
_output_shapes
:         P2

Identity_2Т

Identity_3Identity StatefulPartitionedCall:output:3^StatefulPartitionedCall*
T0*'
_output_shapes
:         P2

Identity_3"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0*(
_construction_contextkEagerRuntime*┐
_input_shapesн
к:         А:         А:         А:         А:         А:         А: : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Q M
(
_output_shapes
:         А
!
_user_specified_name	input_1:QM
(
_output_shapes
:         А
!
_user_specified_name	input_2:QM
(
_output_shapes
:         А
!
_user_specified_name	input_3:QM
(
_output_shapes
:         А
!
_user_specified_name	input_4:QM
(
_output_shapes
:         А
!
_user_specified_name	input_5:QM
(
_output_shapes
:         А
!
_user_specified_name	input_6:

_output_shapes
: "╠L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*й
serving_defaultХ
<
input_11
serving_default_input_1:0         А
<
input_21
serving_default_input_2:0         А
<
input_31
serving_default_input_3:0         А
<
input_41
serving_default_input_4:0         А
<
input_51
serving_default_input_5:0         А
<
input_61
serving_default_input_6:0         АA
encoder1/fcn30
StatefulPartitionedCall:0         PA
encoder2/fcn30
StatefulPartitionedCall:1         P:
tf.__operators__.add_18
StatefulPartitionedCall:2 A
tf.math.truediv_4,
StatefulPartitionedCall:3         tensorflow/serving/predict:╢С
Н╗
layer-0
layer-1
layer-2
layer_with_weights-0
layer-3
layer-4
layer-5
layer_with_weights-1
layer-6
layer_with_weights-2
layer-7
	layer_with_weights-3
	layer-8

layer-9
layer-10
layer-11
layer-12
layer-13
layer-14
layer_with_weights-4
layer-15
layer-16
layer-17
layer-18
layer-19
layer-20
layer_with_weights-5
layer-21
layer-22
layer-23
layer-24
layer-25
layer-26
layer-27
layer-28
layer-29
layer-30
 layer-31
!layer_with_weights-6
!layer-32
"layer_with_weights-7
"layer-33
#layer-34
$layer-35
%layer_with_weights-8
%layer-36
&layer_with_weights-9
&layer-37
'layer-38
(layer-39
)layer-40
*layer-41
+layer-42
,layer-43
-layer-44
.layer-45
/layer-46
0layer-47
1layer-48
2layer-49
3layer-50
4layer-51
5layer-52
6layer-53
7layer_with_weights-10
7layer-54
8layer_with_weights-11
8layer-55
9layer-56
:layer-57
;layer-58
<layer-59
=layer-60
>layer-61
?layer-62
@layer-63
Alayer-64
Blayer-65
Clayer-66
Dlayer-67
Elayer-68
Flayer-69
Glayer-70
Hlayer-71
Ilayer-72
Jlayer-73
Klayer-74
Llayer-75
Mlayer-76
Nlayer-77
Olayer-78
Player-79
Qlayer-80
Rlayer-81
Slayer-82
Tlayer-83
Ulayer-84
Vlayer-85
W	variables
Xregularization_losses
Ytrainable_variables
Z	keras_api
[
signatures
й_default_save_signature
+к&call_and_return_all_conditional_losses
л__call__"щн
_tf_keras_network╠н{"name": "model", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "class_name": "Functional", "config": {"name": "model", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 768]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_2"}, "name": "input_2", "inbound_nodes": []}, {"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 768]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_1"}, "name": "input_1", "inbound_nodes": []}, {"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 768]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_6"}, "name": "input_6", "inbound_nodes": []}, {"class_name": "Dense", "config": {"name": "encoder2/fcn1", "trainable": true, "dtype": "float32", "units": 120, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "encoder2/fcn1", "inbound_nodes": [[["input_2", 0, 0, {}]], [["input_5", 0, 0, {}]]]}, {"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 768]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_3"}, "name": "input_3", "inbound_nodes": []}, {"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 768]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_4"}, "name": "input_4", "inbound_nodes": []}, {"class_name": "Dense", "config": {"name": "encoder1/fcn1", "trainable": true, "dtype": "float32", "units": 100, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "encoder1/fcn1", "inbound_nodes": [[["input_1", 0, 0, {}]], [["input_3", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "encoder1/fcn3_gene", "trainable": true, "dtype": "float32", "units": 100, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "encoder1/fcn3_gene", "inbound_nodes": [[["input_6", 0, 0, {}]], [["input_4", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "encoder2/fcn2/mean", "trainable": true, "dtype": "float32", "units": 96, "activation": "leaky_relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "encoder2/fcn2/mean", "inbound_nodes": [[["encoder2/fcn1", 0, 0, {}]], [["encoder2/fcn1", 1, 0, {}]]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.math.subtract", "trainable": true, "dtype": "float32", "function": "math.subtract"}, "name": "tf.math.subtract", "inbound_nodes": [["encoder1/fcn1", 0, 0, {"y": ["encoder1/fcn3_gene", 0, 0], "name": null}]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.math.subtract_3", "trainable": true, "dtype": "float32", "function": "math.subtract"}, "name": "tf.math.subtract_3", "inbound_nodes": [["encoder1/fcn1", 1, 0, {"y": ["encoder1/fcn3_gene", 1, 0], "name": null}]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.math.square_2", "trainable": true, "dtype": "float32", "function": "math.square"}, "name": "tf.math.square_2", "inbound_nodes": [["input_2", 0, 0, {"name": null}]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.math.square_3", "trainable": true, "dtype": "float32", "function": "math.square"}, "name": "tf.math.square_3", "inbound_nodes": [["input_2", 0, 0, {"name": null}]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.math.square_4", "trainable": true, "dtype": "float32", "function": "math.square"}, "name": "tf.math.square_4", "inbound_nodes": [["encoder2/fcn2/mean", 0, 0, {"name": null}]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.math.square_5", "trainable": true, "dtype": "float32", "function": "math.square"}, "name": "tf.math.square_5", "inbound_nodes": [["encoder2/fcn2/mean", 0, 0, {"name": null}]]}, {"class_name": "Dense", "config": {"name": "encoder1/fcn2/mean", "trainable": true, "dtype": "float32", "units": 100, "activation": "leaky_relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "encoder1/fcn2/mean", "inbound_nodes": [[["tf.math.subtract", 0, 0, {}]], [["tf.math.subtract_3", 0, 0, {}]]]}, {"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 768]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_5"}, "name": "input_5", "inbound_nodes": []}, {"class_name": "TFOpLambda", "config": {"name": "tf.math.reduce_sum", "trainable": true, "dtype": "float32", "function": "math.reduce_sum"}, "name": "tf.math.reduce_sum", "inbound_nodes": [["tf.math.square_2", 0, 0, {"axis": -1, "keepdims": true}]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.math.reduce_sum_1", "trainable": true, "dtype": "float32", "function": "math.reduce_sum"}, "name": "tf.math.reduce_sum_1", "inbound_nodes": [["tf.math.square_3", 0, 0, {"axis": -1, "keepdims": true}]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.math.reduce_sum_2", "trainable": true, "dtype": "float32", "function": "math.reduce_sum"}, "name": "tf.math.reduce_sum_2", "inbound_nodes": [["tf.math.square_4", 0, 0, {"axis": -1, "keepdims": true}]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.math.reduce_sum_3", "trainable": true, "dtype": "float32", "function": "math.reduce_sum"}, "name": "tf.math.reduce_sum_3", "inbound_nodes": [["tf.math.square_5", 0, 0, {"axis": -1, "keepdims": true}]]}, {"class_name": "Dense", "config": {"name": "decoder1/layer_gene", "trainable": true, "dtype": "float32", "units": 100, "activation": "leaky_relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "decoder1/layer_gene", "inbound_nodes": [[["encoder1/fcn2/mean", 0, 0, {}]], [["encoder1/fcn2/mean", 1, 0, {}]]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.math.sqrt", "trainable": true, "dtype": "float32", "function": "math.sqrt"}, "name": "tf.math.sqrt", "inbound_nodes": [["tf.math.reduce_sum", 0, 0, {}]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.math.sqrt_1", "trainable": true, "dtype": "float32", "function": "math.sqrt"}, "name": "tf.math.sqrt_1", "inbound_nodes": [["tf.math.reduce_sum_1", 0, 0, {}]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.math.sqrt_2", "trainable": true, "dtype": "float32", "function": "math.sqrt"}, "name": "tf.math.sqrt_2", "inbound_nodes": [["tf.math.reduce_sum_2", 0, 0, {}]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.math.sqrt_3", "trainable": true, "dtype": "float32", "function": "math.sqrt"}, "name": "tf.math.sqrt_3", "inbound_nodes": [["tf.math.reduce_sum_3", 0, 0, {}]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.__operators__.add", "trainable": true, "dtype": "float32", "function": "__operators__.add"}, "name": "tf.__operators__.add", "inbound_nodes": [["decoder1/layer_gene", 0, 0, {"y": ["encoder1/fcn3_gene", 0, 0], "name": null}]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.__operators__.add_2", "trainable": true, "dtype": "float32", "function": "__operators__.add"}, "name": "tf.__operators__.add_2", "inbound_nodes": [["decoder1/layer_gene", 1, 0, {"y": ["encoder1/fcn3_gene", 1, 0], "name": null}]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.linalg.matmul", "trainable": true, "dtype": "float32", "function": "linalg.matmul"}, "name": "tf.linalg.matmul", "inbound_nodes": [["input_2", 0, 0, {"b": ["input_2", 0, 0], "transpose_b": true}]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.math.multiply_2", "trainable": true, "dtype": "float32", "function": "math.multiply"}, "name": "tf.math.multiply_2", "inbound_nodes": [["tf.math.sqrt", 0, 0, {"y": ["tf.math.sqrt_1", 0, 0], "name": null}]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.linalg.matmul_1", "trainable": true, "dtype": "float32", "function": "linalg.matmul"}, "name": "tf.linalg.matmul_1", "inbound_nodes": [["encoder2/fcn2/mean", 0, 0, {"b": ["encoder2/fcn2/mean", 0, 0], "transpose_b": true}]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.math.multiply_3", "trainable": true, "dtype": "float32", "function": "math.multiply"}, "name": "tf.math.multiply_3", "inbound_nodes": [["tf.math.sqrt_2", 0, 0, {"y": ["tf.math.sqrt_3", 0, 0], "name": null}]]}, {"class_name": "Dense", "config": {"name": "decoder1/fcn1", "trainable": true, "dtype": "float32", "units": 1152, "activation": "leaky_relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "decoder1/fcn1", "inbound_nodes": [[["tf.__operators__.add", 0, 0, {}]], [["tf.__operators__.add_2", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "decoder2/fcn1", "trainable": true, "dtype": "float32", "units": 1152, "activation": "leaky_relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "decoder2/fcn1", "inbound_nodes": [[["encoder2/fcn2/mean", 0, 0, {}]], [["encoder2/fcn2/mean", 1, 0, {}]]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.math.truediv", "trainable": true, "dtype": "float32", "function": "math.truediv"}, "name": "tf.math.truediv", "inbound_nodes": [["tf.linalg.matmul", 0, 0, {"y": ["tf.math.multiply_2", 0, 0], "name": null}]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.math.truediv_1", "trainable": true, "dtype": "float32", "function": "math.truediv"}, "name": "tf.math.truediv_1", "inbound_nodes": [["tf.linalg.matmul_1", 0, 0, {"y": ["tf.math.multiply_3", 0, 0], "name": null}]]}, {"class_name": "Dense", "config": {"name": "decoder1/fcn2", "trainable": true, "dtype": "float32", "units": 768, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "decoder1/fcn2", "inbound_nodes": [[["decoder1/fcn1", 0, 0, {}]], [["decoder1/fcn1", 1, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "decoder2/fcn2", "trainable": true, "dtype": "float32", "units": 768, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "decoder2/fcn2", "inbound_nodes": [[["decoder2/fcn1", 0, 0, {}]], [["decoder2/fcn1", 1, 0, {}]]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.math.multiply_8", "trainable": true, "dtype": "float32", "function": "math.multiply"}, "name": "tf.math.multiply_8", "inbound_nodes": [["input_3", 0, 0, {"y": 50.0, "name": null}]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.math.multiply_9", "trainable": true, "dtype": "float32", "function": "math.multiply"}, "name": "tf.math.multiply_9", "inbound_nodes": [["input_5", 0, 0, {"y": 50.0, "name": null}]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.nn.softmax", "trainable": true, "dtype": "float32", "function": "nn.softmax"}, "name": "tf.nn.softmax", "inbound_nodes": [["tf.math.truediv", 0, 0, {"axis": 1}]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.nn.softmax_1", "trainable": true, "dtype": "float32", "function": "nn.softmax"}, "name": "tf.nn.softmax_1", "inbound_nodes": [["tf.math.truediv_1", 0, 0, {"axis": 1}]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.math.multiply", "trainable": true, "dtype": "float32", "function": "math.multiply"}, "name": "tf.math.multiply", "inbound_nodes": [["input_1", 0, 0, {"y": 50.0, "name": null}]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.math.multiply_1", "trainable": true, "dtype": "float32", "function": "math.multiply"}, "name": "tf.math.multiply_1", "inbound_nodes": [["input_2", 0, 0, {"y": 50.0, "name": null}]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.math.subtract_4", "trainable": true, "dtype": "float32", "function": "math.subtract"}, "name": "tf.math.subtract_4", "inbound_nodes": [["tf.math.multiply_8", 0, 0, {"y": ["decoder1/fcn2", 1, 0], "name": null}]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.math.subtract_5", "trainable": true, "dtype": "float32", "function": "math.subtract"}, "name": "tf.math.subtract_5", "inbound_nodes": [["tf.math.multiply_9", 0, 0, {"y": ["decoder2/fcn2", 1, 0], "name": null}]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.math.truediv_2", "trainable": true, "dtype": "float32", "function": "math.truediv"}, "name": "tf.math.truediv_2", "inbound_nodes": [["tf.nn.softmax", 0, 0, {"y": ["tf.nn.softmax_1", 0, 0], "name": null}]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.math.truediv_3", "trainable": true, "dtype": "float32", "function": "math.truediv"}, "name": "tf.math.truediv_3", "inbound_nodes": [["tf.nn.softmax_1", 0, 0, {"y": ["tf.nn.softmax", 0, 0], "name": null}]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.math.subtract_1", "trainable": true, "dtype": "float32", "function": "math.subtract"}, "name": "tf.math.subtract_1", "inbound_nodes": [["tf.math.multiply", 0, 0, {"y": ["decoder1/fcn2", 0, 0], "name": null}]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.math.subtract_2", "trainable": true, "dtype": "float32", "function": "math.subtract"}, "name": "tf.math.subtract_2", "inbound_nodes": [["tf.math.multiply_1", 0, 0, {"y": ["decoder2/fcn2", 0, 0], "name": null}]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.math.square_8", "trainable": true, "dtype": "float32", "function": "math.square"}, "name": "tf.math.square_8", "inbound_nodes": [["tf.math.subtract_4", 0, 0, {"name": null}]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.math.square_9", "trainable": true, "dtype": "float32", "function": "math.square"}, "name": "tf.math.square_9", "inbound_nodes": [["tf.math.subtract_5", 0, 0, {"name": null}]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.math.log", "trainable": true, "dtype": "float32", "function": "math.log"}, "name": "tf.math.log", "inbound_nodes": [["tf.math.truediv_2", 0, 0, {"name": null}]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.math.log_1", "trainable": true, "dtype": "float32", "function": "math.log"}, "name": "tf.math.log_1", "inbound_nodes": [["tf.math.truediv_3", 0, 0, {"name": null}]]}, {"class_name": "Dense", "config": {"name": "encoder1/fcn3", "trainable": true, "dtype": "float32", "units": 80, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "encoder1/fcn3", "inbound_nodes": [[["encoder1/fcn2/mean", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "encoder2/fcn3", "trainable": true, "dtype": "float32", "units": 80, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "encoder2/fcn3", "inbound_nodes": [[["encoder2/fcn2/mean", 0, 0, {}]]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.math.square", "trainable": true, "dtype": "float32", "function": "math.square"}, "name": "tf.math.square", "inbound_nodes": [["tf.math.subtract_1", 0, 0, {"name": null}]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.math.square_1", "trainable": true, "dtype": "float32", "function": "math.square"}, "name": "tf.math.square_1", "inbound_nodes": [["tf.math.subtract_2", 0, 0, {"name": null}]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.math.reduce_mean_3", "trainable": true, "dtype": "float32", "function": "math.reduce_mean"}, "name": "tf.math.reduce_mean_3", "inbound_nodes": [["tf.math.square_8", 0, 0, {}]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.math.reduce_mean_4", "trainable": true, "dtype": "float32", "function": "math.reduce_mean"}, "name": "tf.math.reduce_mean_4", "inbound_nodes": [["tf.math.square_9", 0, 0, {}]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.math.multiply_4", "trainable": true, "dtype": "float32", "function": "math.multiply"}, "name": "tf.math.multiply_4", "inbound_nodes": [["tf.nn.softmax", 0, 0, {"y": ["tf.math.log", 0, 0], "name": null}]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.math.multiply_5", "trainable": true, "dtype": "float32", "function": "math.multiply"}, "name": "tf.math.multiply_5", "inbound_nodes": [["tf.nn.softmax_1", 0, 0, {"y": ["tf.math.log_1", 0, 0], "name": null}]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.math.square_6", "trainable": true, "dtype": "float32", "function": "math.square"}, "name": "tf.math.square_6", "inbound_nodes": [["encoder1/fcn3", 0, 0, {"name": null}]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.math.square_7", "trainable": true, "dtype": "float32", "function": "math.square"}, "name": "tf.math.square_7", "inbound_nodes": [["encoder2/fcn3", 0, 0, {"name": null}]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.math.reduce_mean", "trainable": true, "dtype": "float32", "function": "math.reduce_mean"}, "name": "tf.math.reduce_mean", "inbound_nodes": [["tf.math.square", 0, 0, {}]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.math.reduce_mean_1", "trainable": true, "dtype": "float32", "function": "math.reduce_mean"}, "name": "tf.math.reduce_mean_1", "inbound_nodes": [["tf.math.square_1", 0, 0, {}]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.__operators__.add_16", "trainable": true, "dtype": "float32", "function": "__operators__.add"}, "name": "tf.__operators__.add_16", "inbound_nodes": [["tf.math.reduce_mean_3", 0, 0, {"y": ["tf.math.reduce_mean_4", 0, 0], "name": null}]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.math.reduce_sum_4", "trainable": true, "dtype": "float32", "function": "math.reduce_sum"}, "name": "tf.math.reduce_sum_4", "inbound_nodes": [["tf.math.multiply_4", 0, 0, {"axis": -1}]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.math.reduce_sum_5", "trainable": true, "dtype": "float32", "function": "math.reduce_sum"}, "name": "tf.math.reduce_sum_5", "inbound_nodes": [["tf.math.multiply_5", 0, 0, {"axis": -1}]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.math.reduce_sum_7", "trainable": true, "dtype": "float32", "function": "math.reduce_sum"}, "name": "tf.math.reduce_sum_7", "inbound_nodes": [["tf.math.square_6", 0, 0, {"axis": -1}]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.math.reduce_sum_8", "trainable": true, "dtype": "float32", "function": "math.reduce_sum"}, "name": "tf.math.reduce_sum_8", "inbound_nodes": [["tf.math.square_7", 0, 0, {"axis": -1}]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.__operators__.add_15", "trainable": true, "dtype": "float32", "function": "__operators__.add"}, "name": "tf.__operators__.add_15", "inbound_nodes": [["tf.math.reduce_mean", 0, 0, {"y": ["tf.math.reduce_mean_1", 0, 0], "name": null}]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.math.multiply_22", "trainable": true, "dtype": "float32", "function": "math.multiply"}, "name": "tf.math.multiply_22", "inbound_nodes": [["_CONSTANT_VALUE", -1, 0.20000000298023224, {"y": ["tf.__operators__.add_16", 0, 0], "name": null}]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.__operators__.add_1", "trainable": true, "dtype": "float32", "function": "__operators__.add"}, "name": "tf.__operators__.add_1", "inbound_nodes": [["tf.math.reduce_sum_4", 0, 0, {"y": ["tf.math.reduce_sum_5", 0, 0], "name": null}]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.math.multiply_6", "trainable": true, "dtype": "float32", "function": "math.multiply"}, "name": "tf.math.multiply_6", "inbound_nodes": [["encoder1/fcn3", 0, 0, {"y": ["encoder2/fcn3", 0, 0]}]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.math.sqrt_4", "trainable": true, "dtype": "float32", "function": "math.sqrt"}, "name": "tf.math.sqrt_4", "inbound_nodes": [["tf.math.reduce_sum_7", 0, 0, {}]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.math.sqrt_5", "trainable": true, "dtype": "float32", "function": "math.sqrt"}, "name": "tf.math.sqrt_5", "inbound_nodes": [["tf.math.reduce_sum_8", 0, 0, {}]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.math.truediv_21", "trainable": true, "dtype": "float32", "function": "math.truediv"}, "name": "tf.math.truediv_21", "inbound_nodes": [["tf.__operators__.add_15", 0, 0, {"y": 2.5, "name": null}]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.math.truediv_22", "trainable": true, "dtype": "float32", "function": "math.truediv"}, "name": "tf.math.truediv_22", "inbound_nodes": [["tf.math.multiply_22", 0, 0, {"y": 2.0, "name": null}]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.math.reduce_mean_2", "trainable": true, "dtype": "float32", "function": "math.reduce_mean"}, "name": "tf.math.reduce_mean_2", "inbound_nodes": [["tf.__operators__.add_1", 0, 0, {}]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.math.reduce_sum_6", "trainable": true, "dtype": "float32", "function": "math.reduce_sum"}, "name": "tf.math.reduce_sum_6", "inbound_nodes": [["tf.math.multiply_6", 0, 0, {"axis": -1}]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.math.multiply_7", "trainable": true, "dtype": "float32", "function": "math.multiply"}, "name": "tf.math.multiply_7", "inbound_nodes": [["tf.math.sqrt_4", 0, 0, {"y": ["tf.math.sqrt_5", 0, 0], "name": null}]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.__operators__.add_17", "trainable": true, "dtype": "float32", "function": "__operators__.add"}, "name": "tf.__operators__.add_17", "inbound_nodes": [["tf.math.truediv_21", 0, 0, {"y": ["tf.math.truediv_22", 0, 0], "name": null}]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.math.multiply_23", "trainable": true, "dtype": "float32", "function": "math.multiply"}, "name": "tf.math.multiply_23", "inbound_nodes": [["tf.math.reduce_mean_2", 0, 0, {"y": 0.5, "name": null}]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.math.truediv_4", "trainable": true, "dtype": "float32", "function": "math.truediv"}, "name": "tf.math.truediv_4", "inbound_nodes": [["tf.math.reduce_sum_6", 0, 0, {"y": ["tf.math.multiply_7", 0, 0], "name": null}]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.__operators__.add_18", "trainable": true, "dtype": "float32", "function": "__operators__.add"}, "name": "tf.__operators__.add_18", "inbound_nodes": [["tf.__operators__.add_17", 0, 0, {"y": ["tf.math.multiply_23", 0, 0], "name": null}]]}], "input_layers": [["input_1", 0, 0], ["input_2", 0, 0], ["input_3", 0, 0], ["input_4", 0, 0], ["input_5", 0, 0], ["input_6", 0, 0]], "output_layers": [["tf.math.truediv_4", 0, 0], ["tf.__operators__.add_18", 0, 0], ["encoder1/fcn3", 0, 0], ["encoder2/fcn3", 0, 0]]}, "shared_object_id": 110, "input_spec": [{"class_name": "InputSpec", "config": {"dtype": null, "shape": {"class_name": "__tuple__", "items": [null, 768]}, "ndim": 2, "max_ndim": null, "min_ndim": null, "axes": {}}}, {"class_name": "InputSpec", "config": {"dtype": null, "shape": {"class_name": "__tuple__", "items": [null, 768]}, "ndim": 2, "max_ndim": null, "min_ndim": null, "axes": {}}}, {"class_name": "InputSpec", "config": {"dtype": null, "shape": {"class_name": "__tuple__", "items": [null, 768]}, "ndim": 2, "max_ndim": null, "min_ndim": null, "axes": {}}}, {"class_name": "InputSpec", "config": {"dtype": null, "shape": {"class_name": "__tuple__", "items": [null, 768]}, "ndim": 2, "max_ndim": null, "min_ndim": null, "axes": {}}}, {"class_name": "InputSpec", "config": {"dtype": null, "shape": {"class_name": "__tuple__", "items": [null, 768]}, "ndim": 2, "max_ndim": null, "min_ndim": null, "axes": {}}}, {"class_name": "InputSpec", "config": {"dtype": null, "shape": {"class_name": "__tuple__", "items": [null, 768]}, "ndim": 2, "max_ndim": null, "min_ndim": null, "axes": {}}}], "build_input_shape": [{"class_name": "TensorShape", "items": [null, 768]}, {"class_name": "TensorShape", "items": [null, 768]}, {"class_name": "TensorShape", "items": [null, 768]}, {"class_name": "TensorShape", "items": [null, 768]}, {"class_name": "TensorShape", "items": [null, 768]}, {"class_name": "TensorShape", "items": [null, 768]}], "is_graph_network": true, "save_spec": [{"class_name": "TypeSpec", "type_spec": "tf.TensorSpec", "serialized": [{"class_name": "TensorShape", "items": [null, 768]}, "float32", "input_1"]}, {"class_name": "TypeSpec", "type_spec": "tf.TensorSpec", "serialized": [{"class_name": "TensorShape", "items": [null, 768]}, "float32", "input_2"]}, {"class_name": "TypeSpec", "type_spec": "tf.TensorSpec", "serialized": [{"class_name": "TensorShape", "items": [null, 768]}, "float32", "input_3"]}, {"class_name": "TypeSpec", "type_spec": "tf.TensorSpec", "serialized": [{"class_name": "TensorShape", "items": [null, 768]}, "float32", "input_4"]}, {"class_name": "TypeSpec", "type_spec": "tf.TensorSpec", "serialized": [{"class_name": "TensorShape", "items": [null, 768]}, "float32", "input_5"]}, {"class_name": "TypeSpec", "type_spec": "tf.TensorSpec", "serialized": [{"class_name": "TensorShape", "items": [null, 768]}, "float32", "input_6"]}], "keras_version": "2.5.0", "backend": "tensorflow", "model_config": {"class_name": "Functional", "config": {"name": "model", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 768]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_2"}, "name": "input_2", "inbound_nodes": [], "shared_object_id": 0}, {"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 768]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_1"}, "name": "input_1", "inbound_nodes": [], "shared_object_id": 1}, {"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 768]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_6"}, "name": "input_6", "inbound_nodes": [], "shared_object_id": 2}, {"class_name": "Dense", "config": {"name": "encoder2/fcn1", "trainable": true, "dtype": "float32", "units": 120, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 3}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 4}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "encoder2/fcn1", "inbound_nodes": [[["input_2", 0, 0, {}]], [["input_5", 0, 0, {}]]], "shared_object_id": 5}, {"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 768]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_3"}, "name": "input_3", "inbound_nodes": [], "shared_object_id": 6}, {"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 768]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_4"}, "name": "input_4", "inbound_nodes": [], "shared_object_id": 7}, {"class_name": "Dense", "config": {"name": "encoder1/fcn1", "trainable": true, "dtype": "float32", "units": 100, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 8}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 9}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "encoder1/fcn1", "inbound_nodes": [[["input_1", 0, 0, {}]], [["input_3", 0, 0, {}]]], "shared_object_id": 10}, {"class_name": "Dense", "config": {"name": "encoder1/fcn3_gene", "trainable": true, "dtype": "float32", "units": 100, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 11}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 12}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "encoder1/fcn3_gene", "inbound_nodes": [[["input_6", 0, 0, {}]], [["input_4", 0, 0, {}]]], "shared_object_id": 13}, {"class_name": "Dense", "config": {"name": "encoder2/fcn2/mean", "trainable": true, "dtype": "float32", "units": 96, "activation": "leaky_relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 14}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 15}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "encoder2/fcn2/mean", "inbound_nodes": [[["encoder2/fcn1", 0, 0, {}]], [["encoder2/fcn1", 1, 0, {}]]], "shared_object_id": 16}, {"class_name": "TFOpLambda", "config": {"name": "tf.math.subtract", "trainable": true, "dtype": "float32", "function": "math.subtract"}, "name": "tf.math.subtract", "inbound_nodes": [["encoder1/fcn1", 0, 0, {"y": ["encoder1/fcn3_gene", 0, 0], "name": null}]], "shared_object_id": 17}, {"class_name": "TFOpLambda", "config": {"name": "tf.math.subtract_3", "trainable": true, "dtype": "float32", "function": "math.subtract"}, "name": "tf.math.subtract_3", "inbound_nodes": [["encoder1/fcn1", 1, 0, {"y": ["encoder1/fcn3_gene", 1, 0], "name": null}]], "shared_object_id": 18}, {"class_name": "TFOpLambda", "config": {"name": "tf.math.square_2", "trainable": true, "dtype": "float32", "function": "math.square"}, "name": "tf.math.square_2", "inbound_nodes": [["input_2", 0, 0, {"name": null}]], "shared_object_id": 19}, {"class_name": "TFOpLambda", "config": {"name": "tf.math.square_3", "trainable": true, "dtype": "float32", "function": "math.square"}, "name": "tf.math.square_3", "inbound_nodes": [["input_2", 0, 0, {"name": null}]], "shared_object_id": 20}, {"class_name": "TFOpLambda", "config": {"name": "tf.math.square_4", "trainable": true, "dtype": "float32", "function": "math.square"}, "name": "tf.math.square_4", "inbound_nodes": [["encoder2/fcn2/mean", 0, 0, {"name": null}]], "shared_object_id": 21}, {"class_name": "TFOpLambda", "config": {"name": "tf.math.square_5", "trainable": true, "dtype": "float32", "function": "math.square"}, "name": "tf.math.square_5", "inbound_nodes": [["encoder2/fcn2/mean", 0, 0, {"name": null}]], "shared_object_id": 22}, {"class_name": "Dense", "config": {"name": "encoder1/fcn2/mean", "trainable": true, "dtype": "float32", "units": 100, "activation": "leaky_relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 23}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 24}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "encoder1/fcn2/mean", "inbound_nodes": [[["tf.math.subtract", 0, 0, {}]], [["tf.math.subtract_3", 0, 0, {}]]], "shared_object_id": 25}, {"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 768]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_5"}, "name": "input_5", "inbound_nodes": [], "shared_object_id": 26}, {"class_name": "TFOpLambda", "config": {"name": "tf.math.reduce_sum", "trainable": true, "dtype": "float32", "function": "math.reduce_sum"}, "name": "tf.math.reduce_sum", "inbound_nodes": [["tf.math.square_2", 0, 0, {"axis": -1, "keepdims": true}]], "shared_object_id": 27}, {"class_name": "TFOpLambda", "config": {"name": "tf.math.reduce_sum_1", "trainable": true, "dtype": "float32", "function": "math.reduce_sum"}, "name": "tf.math.reduce_sum_1", "inbound_nodes": [["tf.math.square_3", 0, 0, {"axis": -1, "keepdims": true}]], "shared_object_id": 28}, {"class_name": "TFOpLambda", "config": {"name": "tf.math.reduce_sum_2", "trainable": true, "dtype": "float32", "function": "math.reduce_sum"}, "name": "tf.math.reduce_sum_2", "inbound_nodes": [["tf.math.square_4", 0, 0, {"axis": -1, "keepdims": true}]], "shared_object_id": 29}, {"class_name": "TFOpLambda", "config": {"name": "tf.math.reduce_sum_3", "trainable": true, "dtype": "float32", "function": "math.reduce_sum"}, "name": "tf.math.reduce_sum_3", "inbound_nodes": [["tf.math.square_5", 0, 0, {"axis": -1, "keepdims": true}]], "shared_object_id": 30}, {"class_name": "Dense", "config": {"name": "decoder1/layer_gene", "trainable": true, "dtype": "float32", "units": 100, "activation": "leaky_relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 31}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 32}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "decoder1/layer_gene", "inbound_nodes": [[["encoder1/fcn2/mean", 0, 0, {}]], [["encoder1/fcn2/mean", 1, 0, {}]]], "shared_object_id": 33}, {"class_name": "TFOpLambda", "config": {"name": "tf.math.sqrt", "trainable": true, "dtype": "float32", "function": "math.sqrt"}, "name": "tf.math.sqrt", "inbound_nodes": [["tf.math.reduce_sum", 0, 0, {}]], "shared_object_id": 34}, {"class_name": "TFOpLambda", "config": {"name": "tf.math.sqrt_1", "trainable": true, "dtype": "float32", "function": "math.sqrt"}, "name": "tf.math.sqrt_1", "inbound_nodes": [["tf.math.reduce_sum_1", 0, 0, {}]], "shared_object_id": 35}, {"class_name": "TFOpLambda", "config": {"name": "tf.math.sqrt_2", "trainable": true, "dtype": "float32", "function": "math.sqrt"}, "name": "tf.math.sqrt_2", "inbound_nodes": [["tf.math.reduce_sum_2", 0, 0, {}]], "shared_object_id": 36}, {"class_name": "TFOpLambda", "config": {"name": "tf.math.sqrt_3", "trainable": true, "dtype": "float32", "function": "math.sqrt"}, "name": "tf.math.sqrt_3", "inbound_nodes": [["tf.math.reduce_sum_3", 0, 0, {}]], "shared_object_id": 37}, {"class_name": "TFOpLambda", "config": {"name": "tf.__operators__.add", "trainable": true, "dtype": "float32", "function": "__operators__.add"}, "name": "tf.__operators__.add", "inbound_nodes": [["decoder1/layer_gene", 0, 0, {"y": ["encoder1/fcn3_gene", 0, 0], "name": null}]], "shared_object_id": 38}, {"class_name": "TFOpLambda", "config": {"name": "tf.__operators__.add_2", "trainable": true, "dtype": "float32", "function": "__operators__.add"}, "name": "tf.__operators__.add_2", "inbound_nodes": [["decoder1/layer_gene", 1, 0, {"y": ["encoder1/fcn3_gene", 1, 0], "name": null}]], "shared_object_id": 39}, {"class_name": "TFOpLambda", "config": {"name": "tf.linalg.matmul", "trainable": true, "dtype": "float32", "function": "linalg.matmul"}, "name": "tf.linalg.matmul", "inbound_nodes": [["input_2", 0, 0, {"b": ["input_2", 0, 0], "transpose_b": true}]], "shared_object_id": 40}, {"class_name": "TFOpLambda", "config": {"name": "tf.math.multiply_2", "trainable": true, "dtype": "float32", "function": "math.multiply"}, "name": "tf.math.multiply_2", "inbound_nodes": [["tf.math.sqrt", 0, 0, {"y": ["tf.math.sqrt_1", 0, 0], "name": null}]], "shared_object_id": 41}, {"class_name": "TFOpLambda", "config": {"name": "tf.linalg.matmul_1", "trainable": true, "dtype": "float32", "function": "linalg.matmul"}, "name": "tf.linalg.matmul_1", "inbound_nodes": [["encoder2/fcn2/mean", 0, 0, {"b": ["encoder2/fcn2/mean", 0, 0], "transpose_b": true}]], "shared_object_id": 42}, {"class_name": "TFOpLambda", "config": {"name": "tf.math.multiply_3", "trainable": true, "dtype": "float32", "function": "math.multiply"}, "name": "tf.math.multiply_3", "inbound_nodes": [["tf.math.sqrt_2", 0, 0, {"y": ["tf.math.sqrt_3", 0, 0], "name": null}]], "shared_object_id": 43}, {"class_name": "Dense", "config": {"name": "decoder1/fcn1", "trainable": true, "dtype": "float32", "units": 1152, "activation": "leaky_relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 44}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 45}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "decoder1/fcn1", "inbound_nodes": [[["tf.__operators__.add", 0, 0, {}]], [["tf.__operators__.add_2", 0, 0, {}]]], "shared_object_id": 46}, {"class_name": "Dense", "config": {"name": "decoder2/fcn1", "trainable": true, "dtype": "float32", "units": 1152, "activation": "leaky_relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 47}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 48}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "decoder2/fcn1", "inbound_nodes": [[["encoder2/fcn2/mean", 0, 0, {}]], [["encoder2/fcn2/mean", 1, 0, {}]]], "shared_object_id": 49}, {"class_name": "TFOpLambda", "config": {"name": "tf.math.truediv", "trainable": true, "dtype": "float32", "function": "math.truediv"}, "name": "tf.math.truediv", "inbound_nodes": [["tf.linalg.matmul", 0, 0, {"y": ["tf.math.multiply_2", 0, 0], "name": null}]], "shared_object_id": 50}, {"class_name": "TFOpLambda", "config": {"name": "tf.math.truediv_1", "trainable": true, "dtype": "float32", "function": "math.truediv"}, "name": "tf.math.truediv_1", "inbound_nodes": [["tf.linalg.matmul_1", 0, 0, {"y": ["tf.math.multiply_3", 0, 0], "name": null}]], "shared_object_id": 51}, {"class_name": "Dense", "config": {"name": "decoder1/fcn2", "trainable": true, "dtype": "float32", "units": 768, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 52}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 53}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "decoder1/fcn2", "inbound_nodes": [[["decoder1/fcn1", 0, 0, {}]], [["decoder1/fcn1", 1, 0, {}]]], "shared_object_id": 54}, {"class_name": "Dense", "config": {"name": "decoder2/fcn2", "trainable": true, "dtype": "float32", "units": 768, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 55}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 56}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "decoder2/fcn2", "inbound_nodes": [[["decoder2/fcn1", 0, 0, {}]], [["decoder2/fcn1", 1, 0, {}]]], "shared_object_id": 57}, {"class_name": "TFOpLambda", "config": {"name": "tf.math.multiply_8", "trainable": true, "dtype": "float32", "function": "math.multiply"}, "name": "tf.math.multiply_8", "inbound_nodes": [["input_3", 0, 0, {"y": 50.0, "name": null}]], "shared_object_id": 58}, {"class_name": "TFOpLambda", "config": {"name": "tf.math.multiply_9", "trainable": true, "dtype": "float32", "function": "math.multiply"}, "name": "tf.math.multiply_9", "inbound_nodes": [["input_5", 0, 0, {"y": 50.0, "name": null}]], "shared_object_id": 59}, {"class_name": "TFOpLambda", "config": {"name": "tf.nn.softmax", "trainable": true, "dtype": "float32", "function": "nn.softmax"}, "name": "tf.nn.softmax", "inbound_nodes": [["tf.math.truediv", 0, 0, {"axis": 1}]], "shared_object_id": 60}, {"class_name": "TFOpLambda", "config": {"name": "tf.nn.softmax_1", "trainable": true, "dtype": "float32", "function": "nn.softmax"}, "name": "tf.nn.softmax_1", "inbound_nodes": [["tf.math.truediv_1", 0, 0, {"axis": 1}]], "shared_object_id": 61}, {"class_name": "TFOpLambda", "config": {"name": "tf.math.multiply", "trainable": true, "dtype": "float32", "function": "math.multiply"}, "name": "tf.math.multiply", "inbound_nodes": [["input_1", 0, 0, {"y": 50.0, "name": null}]], "shared_object_id": 62}, {"class_name": "TFOpLambda", "config": {"name": "tf.math.multiply_1", "trainable": true, "dtype": "float32", "function": "math.multiply"}, "name": "tf.math.multiply_1", "inbound_nodes": [["input_2", 0, 0, {"y": 50.0, "name": null}]], "shared_object_id": 63}, {"class_name": "TFOpLambda", "config": {"name": "tf.math.subtract_4", "trainable": true, "dtype": "float32", "function": "math.subtract"}, "name": "tf.math.subtract_4", "inbound_nodes": [["tf.math.multiply_8", 0, 0, {"y": ["decoder1/fcn2", 1, 0], "name": null}]], "shared_object_id": 64}, {"class_name": "TFOpLambda", "config": {"name": "tf.math.subtract_5", "trainable": true, "dtype": "float32", "function": "math.subtract"}, "name": "tf.math.subtract_5", "inbound_nodes": [["tf.math.multiply_9", 0, 0, {"y": ["decoder2/fcn2", 1, 0], "name": null}]], "shared_object_id": 65}, {"class_name": "TFOpLambda", "config": {"name": "tf.math.truediv_2", "trainable": true, "dtype": "float32", "function": "math.truediv"}, "name": "tf.math.truediv_2", "inbound_nodes": [["tf.nn.softmax", 0, 0, {"y": ["tf.nn.softmax_1", 0, 0], "name": null}]], "shared_object_id": 66}, {"class_name": "TFOpLambda", "config": {"name": "tf.math.truediv_3", "trainable": true, "dtype": "float32", "function": "math.truediv"}, "name": "tf.math.truediv_3", "inbound_nodes": [["tf.nn.softmax_1", 0, 0, {"y": ["tf.nn.softmax", 0, 0], "name": null}]], "shared_object_id": 67}, {"class_name": "TFOpLambda", "config": {"name": "tf.math.subtract_1", "trainable": true, "dtype": "float32", "function": "math.subtract"}, "name": "tf.math.subtract_1", "inbound_nodes": [["tf.math.multiply", 0, 0, {"y": ["decoder1/fcn2", 0, 0], "name": null}]], "shared_object_id": 68}, {"class_name": "TFOpLambda", "config": {"name": "tf.math.subtract_2", "trainable": true, "dtype": "float32", "function": "math.subtract"}, "name": "tf.math.subtract_2", "inbound_nodes": [["tf.math.multiply_1", 0, 0, {"y": ["decoder2/fcn2", 0, 0], "name": null}]], "shared_object_id": 69}, {"class_name": "TFOpLambda", "config": {"name": "tf.math.square_8", "trainable": true, "dtype": "float32", "function": "math.square"}, "name": "tf.math.square_8", "inbound_nodes": [["tf.math.subtract_4", 0, 0, {"name": null}]], "shared_object_id": 70}, {"class_name": "TFOpLambda", "config": {"name": "tf.math.square_9", "trainable": true, "dtype": "float32", "function": "math.square"}, "name": "tf.math.square_9", "inbound_nodes": [["tf.math.subtract_5", 0, 0, {"name": null}]], "shared_object_id": 71}, {"class_name": "TFOpLambda", "config": {"name": "tf.math.log", "trainable": true, "dtype": "float32", "function": "math.log"}, "name": "tf.math.log", "inbound_nodes": [["tf.math.truediv_2", 0, 0, {"name": null}]], "shared_object_id": 72}, {"class_name": "TFOpLambda", "config": {"name": "tf.math.log_1", "trainable": true, "dtype": "float32", "function": "math.log"}, "name": "tf.math.log_1", "inbound_nodes": [["tf.math.truediv_3", 0, 0, {"name": null}]], "shared_object_id": 73}, {"class_name": "Dense", "config": {"name": "encoder1/fcn3", "trainable": true, "dtype": "float32", "units": 80, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 74}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 75}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "encoder1/fcn3", "inbound_nodes": [[["encoder1/fcn2/mean", 0, 0, {}]]], "shared_object_id": 76}, {"class_name": "Dense", "config": {"name": "encoder2/fcn3", "trainable": true, "dtype": "float32", "units": 80, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 77}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 78}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "encoder2/fcn3", "inbound_nodes": [[["encoder2/fcn2/mean", 0, 0, {}]]], "shared_object_id": 79}, {"class_name": "TFOpLambda", "config": {"name": "tf.math.square", "trainable": true, "dtype": "float32", "function": "math.square"}, "name": "tf.math.square", "inbound_nodes": [["tf.math.subtract_1", 0, 0, {"name": null}]], "shared_object_id": 80}, {"class_name": "TFOpLambda", "config": {"name": "tf.math.square_1", "trainable": true, "dtype": "float32", "function": "math.square"}, "name": "tf.math.square_1", "inbound_nodes": [["tf.math.subtract_2", 0, 0, {"name": null}]], "shared_object_id": 81}, {"class_name": "TFOpLambda", "config": {"name": "tf.math.reduce_mean_3", "trainable": true, "dtype": "float32", "function": "math.reduce_mean"}, "name": "tf.math.reduce_mean_3", "inbound_nodes": [["tf.math.square_8", 0, 0, {}]], "shared_object_id": 82}, {"class_name": "TFOpLambda", "config": {"name": "tf.math.reduce_mean_4", "trainable": true, "dtype": "float32", "function": "math.reduce_mean"}, "name": "tf.math.reduce_mean_4", "inbound_nodes": [["tf.math.square_9", 0, 0, {}]], "shared_object_id": 83}, {"class_name": "TFOpLambda", "config": {"name": "tf.math.multiply_4", "trainable": true, "dtype": "float32", "function": "math.multiply"}, "name": "tf.math.multiply_4", "inbound_nodes": [["tf.nn.softmax", 0, 0, {"y": ["tf.math.log", 0, 0], "name": null}]], "shared_object_id": 84}, {"class_name": "TFOpLambda", "config": {"name": "tf.math.multiply_5", "trainable": true, "dtype": "float32", "function": "math.multiply"}, "name": "tf.math.multiply_5", "inbound_nodes": [["tf.nn.softmax_1", 0, 0, {"y": ["tf.math.log_1", 0, 0], "name": null}]], "shared_object_id": 85}, {"class_name": "TFOpLambda", "config": {"name": "tf.math.square_6", "trainable": true, "dtype": "float32", "function": "math.square"}, "name": "tf.math.square_6", "inbound_nodes": [["encoder1/fcn3", 0, 0, {"name": null}]], "shared_object_id": 86}, {"class_name": "TFOpLambda", "config": {"name": "tf.math.square_7", "trainable": true, "dtype": "float32", "function": "math.square"}, "name": "tf.math.square_7", "inbound_nodes": [["encoder2/fcn3", 0, 0, {"name": null}]], "shared_object_id": 87}, {"class_name": "TFOpLambda", "config": {"name": "tf.math.reduce_mean", "trainable": true, "dtype": "float32", "function": "math.reduce_mean"}, "name": "tf.math.reduce_mean", "inbound_nodes": [["tf.math.square", 0, 0, {}]], "shared_object_id": 88}, {"class_name": "TFOpLambda", "config": {"name": "tf.math.reduce_mean_1", "trainable": true, "dtype": "float32", "function": "math.reduce_mean"}, "name": "tf.math.reduce_mean_1", "inbound_nodes": [["tf.math.square_1", 0, 0, {}]], "shared_object_id": 89}, {"class_name": "TFOpLambda", "config": {"name": "tf.__operators__.add_16", "trainable": true, "dtype": "float32", "function": "__operators__.add"}, "name": "tf.__operators__.add_16", "inbound_nodes": [["tf.math.reduce_mean_3", 0, 0, {"y": ["tf.math.reduce_mean_4", 0, 0], "name": null}]], "shared_object_id": 90}, {"class_name": "TFOpLambda", "config": {"name": "tf.math.reduce_sum_4", "trainable": true, "dtype": "float32", "function": "math.reduce_sum"}, "name": "tf.math.reduce_sum_4", "inbound_nodes": [["tf.math.multiply_4", 0, 0, {"axis": -1}]], "shared_object_id": 91}, {"class_name": "TFOpLambda", "config": {"name": "tf.math.reduce_sum_5", "trainable": true, "dtype": "float32", "function": "math.reduce_sum"}, "name": "tf.math.reduce_sum_5", "inbound_nodes": [["tf.math.multiply_5", 0, 0, {"axis": -1}]], "shared_object_id": 92}, {"class_name": "TFOpLambda", "config": {"name": "tf.math.reduce_sum_7", "trainable": true, "dtype": "float32", "function": "math.reduce_sum"}, "name": "tf.math.reduce_sum_7", "inbound_nodes": [["tf.math.square_6", 0, 0, {"axis": -1}]], "shared_object_id": 93}, {"class_name": "TFOpLambda", "config": {"name": "tf.math.reduce_sum_8", "trainable": true, "dtype": "float32", "function": "math.reduce_sum"}, "name": "tf.math.reduce_sum_8", "inbound_nodes": [["tf.math.square_7", 0, 0, {"axis": -1}]], "shared_object_id": 94}, {"class_name": "TFOpLambda", "config": {"name": "tf.__operators__.add_15", "trainable": true, "dtype": "float32", "function": "__operators__.add"}, "name": "tf.__operators__.add_15", "inbound_nodes": [["tf.math.reduce_mean", 0, 0, {"y": ["tf.math.reduce_mean_1", 0, 0], "name": null}]], "shared_object_id": 95}, {"class_name": "TFOpLambda", "config": {"name": "tf.math.multiply_22", "trainable": true, "dtype": "float32", "function": "math.multiply"}, "name": "tf.math.multiply_22", "inbound_nodes": [["_CONSTANT_VALUE", -1, 0.20000000298023224, {"y": ["tf.__operators__.add_16", 0, 0], "name": null}]], "shared_object_id": 96}, {"class_name": "TFOpLambda", "config": {"name": "tf.__operators__.add_1", "trainable": true, "dtype": "float32", "function": "__operators__.add"}, "name": "tf.__operators__.add_1", "inbound_nodes": [["tf.math.reduce_sum_4", 0, 0, {"y": ["tf.math.reduce_sum_5", 0, 0], "name": null}]], "shared_object_id": 97}, {"class_name": "TFOpLambda", "config": {"name": "tf.math.multiply_6", "trainable": true, "dtype": "float32", "function": "math.multiply"}, "name": "tf.math.multiply_6", "inbound_nodes": [["encoder1/fcn3", 0, 0, {"y": ["encoder2/fcn3", 0, 0]}]], "shared_object_id": 98}, {"class_name": "TFOpLambda", "config": {"name": "tf.math.sqrt_4", "trainable": true, "dtype": "float32", "function": "math.sqrt"}, "name": "tf.math.sqrt_4", "inbound_nodes": [["tf.math.reduce_sum_7", 0, 0, {}]], "shared_object_id": 99}, {"class_name": "TFOpLambda", "config": {"name": "tf.math.sqrt_5", "trainable": true, "dtype": "float32", "function": "math.sqrt"}, "name": "tf.math.sqrt_5", "inbound_nodes": [["tf.math.reduce_sum_8", 0, 0, {}]], "shared_object_id": 100}, {"class_name": "TFOpLambda", "config": {"name": "tf.math.truediv_21", "trainable": true, "dtype": "float32", "function": "math.truediv"}, "name": "tf.math.truediv_21", "inbound_nodes": [["tf.__operators__.add_15", 0, 0, {"y": 2.5, "name": null}]], "shared_object_id": 101}, {"class_name": "TFOpLambda", "config": {"name": "tf.math.truediv_22", "trainable": true, "dtype": "float32", "function": "math.truediv"}, "name": "tf.math.truediv_22", "inbound_nodes": [["tf.math.multiply_22", 0, 0, {"y": 2.0, "name": null}]], "shared_object_id": 102}, {"class_name": "TFOpLambda", "config": {"name": "tf.math.reduce_mean_2", "trainable": true, "dtype": "float32", "function": "math.reduce_mean"}, "name": "tf.math.reduce_mean_2", "inbound_nodes": [["tf.__operators__.add_1", 0, 0, {}]], "shared_object_id": 103}, {"class_name": "TFOpLambda", "config": {"name": "tf.math.reduce_sum_6", "trainable": true, "dtype": "float32", "function": "math.reduce_sum"}, "name": "tf.math.reduce_sum_6", "inbound_nodes": [["tf.math.multiply_6", 0, 0, {"axis": -1}]], "shared_object_id": 104}, {"class_name": "TFOpLambda", "config": {"name": "tf.math.multiply_7", "trainable": true, "dtype": "float32", "function": "math.multiply"}, "name": "tf.math.multiply_7", "inbound_nodes": [["tf.math.sqrt_4", 0, 0, {"y": ["tf.math.sqrt_5", 0, 0], "name": null}]], "shared_object_id": 105}, {"class_name": "TFOpLambda", "config": {"name": "tf.__operators__.add_17", "trainable": true, "dtype": "float32", "function": "__operators__.add"}, "name": "tf.__operators__.add_17", "inbound_nodes": [["tf.math.truediv_21", 0, 0, {"y": ["tf.math.truediv_22", 0, 0], "name": null}]], "shared_object_id": 106}, {"class_name": "TFOpLambda", "config": {"name": "tf.math.multiply_23", "trainable": true, "dtype": "float32", "function": "math.multiply"}, "name": "tf.math.multiply_23", "inbound_nodes": [["tf.math.reduce_mean_2", 0, 0, {"y": 0.5, "name": null}]], "shared_object_id": 107}, {"class_name": "TFOpLambda", "config": {"name": "tf.math.truediv_4", "trainable": true, "dtype": "float32", "function": "math.truediv"}, "name": "tf.math.truediv_4", "inbound_nodes": [["tf.math.reduce_sum_6", 0, 0, {"y": ["tf.math.multiply_7", 0, 0], "name": null}]], "shared_object_id": 108}, {"class_name": "TFOpLambda", "config": {"name": "tf.__operators__.add_18", "trainable": true, "dtype": "float32", "function": "__operators__.add"}, "name": "tf.__operators__.add_18", "inbound_nodes": [["tf.__operators__.add_17", 0, 0, {"y": ["tf.math.multiply_23", 0, 0], "name": null}]], "shared_object_id": 109}], "input_layers": [["input_1", 0, 0], ["input_2", 0, 0], ["input_3", 0, 0], ["input_4", 0, 0], ["input_5", 0, 0], ["input_6", 0, 0]], "output_layers": [["tf.math.truediv_4", 0, 0], ["tf.__operators__.add_18", 0, 0], ["encoder1/fcn3", 0, 0], ["encoder2/fcn3", 0, 0]]}}}
э"ъ
_tf_keras_input_layer╩{"class_name": "InputLayer", "name": "input_2", "dtype": "float32", "sparse": false, "ragged": false, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 768]}, "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 768]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_2"}}
э"ъ
_tf_keras_input_layer╩{"class_name": "InputLayer", "name": "input_1", "dtype": "float32", "sparse": false, "ragged": false, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 768]}, "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 768]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_1"}}
э"ъ
_tf_keras_input_layer╩{"class_name": "InputLayer", "name": "input_6", "dtype": "float32", "sparse": false, "ragged": false, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 768]}, "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 768]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_6"}}
д	

\kernel
]bias
^	variables
_regularization_losses
`trainable_variables
a	keras_api
+м&call_and_return_all_conditional_losses
н__call__"¤
_tf_keras_layerу{"name": "encoder2/fcn1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Dense", "config": {"name": "encoder2/fcn1", "trainable": true, "dtype": "float32", "units": 120, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 3}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 4}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "inbound_nodes": [[["input_2", 0, 0, {}]], [["input_5", 0, 0, {}]]], "shared_object_id": 5, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 768}}, "shared_object_id": 117}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 768]}}
э"ъ
_tf_keras_input_layer╩{"class_name": "InputLayer", "name": "input_3", "dtype": "float32", "sparse": false, "ragged": false, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 768]}, "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 768]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_3"}}
э"ъ
_tf_keras_input_layer╩{"class_name": "InputLayer", "name": "input_4", "dtype": "float32", "sparse": false, "ragged": false, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 768]}, "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 768]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_4"}}
е	

bkernel
cbias
d	variables
eregularization_losses
ftrainable_variables
g	keras_api
+о&call_and_return_all_conditional_losses
п__call__"■
_tf_keras_layerф{"name": "encoder1/fcn1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Dense", "config": {"name": "encoder1/fcn1", "trainable": true, "dtype": "float32", "units": 100, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 8}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 9}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "inbound_nodes": [[["input_1", 0, 0, {}]], [["input_3", 0, 0, {}]]], "shared_object_id": 10, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 768}}, "shared_object_id": 118}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 768]}}
│	

hkernel
ibias
j	variables
kregularization_losses
ltrainable_variables
m	keras_api
+░&call_and_return_all_conditional_losses
▒__call__"М
_tf_keras_layerЄ{"name": "encoder1/fcn3_gene", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Dense", "config": {"name": "encoder1/fcn3_gene", "trainable": true, "dtype": "float32", "units": 100, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 11}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 12}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "inbound_nodes": [[["input_6", 0, 0, {}]], [["input_4", 0, 0, {}]]], "shared_object_id": 13, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 768}}, "shared_object_id": 119}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 768]}}
┬	

nkernel
obias
p	variables
qregularization_losses
rtrainable_variables
s	keras_api
+▓&call_and_return_all_conditional_losses
│__call__"Ы
_tf_keras_layerБ{"name": "encoder2/fcn2/mean", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Dense", "config": {"name": "encoder2/fcn2/mean", "trainable": true, "dtype": "float32", "units": 96, "activation": "leaky_relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 14}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 15}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "inbound_nodes": [[["encoder2/fcn1", 0, 0, {}]], [["encoder2/fcn1", 1, 0, {}]]], "shared_object_id": 16, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 120}}, "shared_object_id": 120}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 120]}}
▌
t	keras_api"╦
_tf_keras_layer▒{"name": "tf.math.subtract", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": true, "class_name": "TFOpLambda", "config": {"name": "tf.math.subtract", "trainable": true, "dtype": "float32", "function": "math.subtract"}, "inbound_nodes": [["encoder1/fcn1", 0, 0, {"y": ["encoder1/fcn3_gene", 0, 0], "name": null}]], "shared_object_id": 17}
с
u	keras_api"╧
_tf_keras_layer╡{"name": "tf.math.subtract_3", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": true, "class_name": "TFOpLambda", "config": {"name": "tf.math.subtract_3", "trainable": true, "dtype": "float32", "function": "math.subtract"}, "inbound_nodes": [["encoder1/fcn1", 1, 0, {"y": ["encoder1/fcn3_gene", 1, 0], "name": null}]], "shared_object_id": 18}
▓
v	keras_api"а
_tf_keras_layerЖ{"name": "tf.math.square_2", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": true, "class_name": "TFOpLambda", "config": {"name": "tf.math.square_2", "trainable": true, "dtype": "float32", "function": "math.square"}, "inbound_nodes": [["input_2", 0, 0, {"name": null}]], "shared_object_id": 19}
▓
w	keras_api"а
_tf_keras_layerЖ{"name": "tf.math.square_3", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": true, "class_name": "TFOpLambda", "config": {"name": "tf.math.square_3", "trainable": true, "dtype": "float32", "function": "math.square"}, "inbound_nodes": [["input_2", 0, 0, {"name": null}]], "shared_object_id": 20}
╜
x	keras_api"л
_tf_keras_layerС{"name": "tf.math.square_4", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": true, "class_name": "TFOpLambda", "config": {"name": "tf.math.square_4", "trainable": true, "dtype": "float32", "function": "math.square"}, "inbound_nodes": [["encoder2/fcn2/mean", 0, 0, {"name": null}]], "shared_object_id": 21}
╜
y	keras_api"л
_tf_keras_layerС{"name": "tf.math.square_5", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": true, "class_name": "TFOpLambda", "config": {"name": "tf.math.square_5", "trainable": true, "dtype": "float32", "function": "math.square"}, "inbound_nodes": [["encoder2/fcn2/mean", 0, 0, {"name": null}]], "shared_object_id": 22}
╦	

zkernel
{bias
|	variables
}regularization_losses
~trainable_variables
	keras_api
+┤&call_and_return_all_conditional_losses
╡__call__"д
_tf_keras_layerК{"name": "encoder1/fcn2/mean", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Dense", "config": {"name": "encoder1/fcn2/mean", "trainable": true, "dtype": "float32", "units": 100, "activation": "leaky_relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 23}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 24}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "inbound_nodes": [[["tf.math.subtract", 0, 0, {}]], [["tf.math.subtract_3", 0, 0, {}]]], "shared_object_id": 25, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 100}}, "shared_object_id": 121}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 100]}}
э"ъ
_tf_keras_input_layer╩{"class_name": "InputLayer", "name": "input_5", "dtype": "float32", "sparse": false, "ragged": false, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 768]}, "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 768]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_5"}}
╘
А	keras_api"┴
_tf_keras_layerз{"name": "tf.math.reduce_sum", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": true, "class_name": "TFOpLambda", "config": {"name": "tf.math.reduce_sum", "trainable": true, "dtype": "float32", "function": "math.reduce_sum"}, "inbound_nodes": [["tf.math.square_2", 0, 0, {"axis": -1, "keepdims": true}]], "shared_object_id": 27}
╪
Б	keras_api"┼
_tf_keras_layerл{"name": "tf.math.reduce_sum_1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": true, "class_name": "TFOpLambda", "config": {"name": "tf.math.reduce_sum_1", "trainable": true, "dtype": "float32", "function": "math.reduce_sum"}, "inbound_nodes": [["tf.math.square_3", 0, 0, {"axis": -1, "keepdims": true}]], "shared_object_id": 28}
╪
В	keras_api"┼
_tf_keras_layerл{"name": "tf.math.reduce_sum_2", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": true, "class_name": "TFOpLambda", "config": {"name": "tf.math.reduce_sum_2", "trainable": true, "dtype": "float32", "function": "math.reduce_sum"}, "inbound_nodes": [["tf.math.square_4", 0, 0, {"axis": -1, "keepdims": true}]], "shared_object_id": 29}
╪
Г	keras_api"┼
_tf_keras_layerл{"name": "tf.math.reduce_sum_3", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": true, "class_name": "TFOpLambda", "config": {"name": "tf.math.reduce_sum_3", "trainable": true, "dtype": "float32", "function": "math.reduce_sum"}, "inbound_nodes": [["tf.math.square_5", 0, 0, {"axis": -1, "keepdims": true}]], "shared_object_id": 30}
╒	
Дkernel
	Еbias
Ж	variables
Зregularization_losses
Иtrainable_variables
Й	keras_api
+╢&call_and_return_all_conditional_losses
╖__call__"и
_tf_keras_layerО{"name": "decoder1/layer_gene", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Dense", "config": {"name": "decoder1/layer_gene", "trainable": true, "dtype": "float32", "units": 100, "activation": "leaky_relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 31}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 32}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "inbound_nodes": [[["encoder1/fcn2/mean", 0, 0, {}]], [["encoder1/fcn2/mean", 1, 0, {}]]], "shared_object_id": 33, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 100}}, "shared_object_id": 122}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 100]}}
и
К	keras_api"Х
_tf_keras_layer√{"name": "tf.math.sqrt", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": true, "class_name": "TFOpLambda", "config": {"name": "tf.math.sqrt", "trainable": true, "dtype": "float32", "function": "math.sqrt"}, "inbound_nodes": [["tf.math.reduce_sum", 0, 0, {}]], "shared_object_id": 34}
о
Л	keras_api"Ы
_tf_keras_layerБ{"name": "tf.math.sqrt_1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": true, "class_name": "TFOpLambda", "config": {"name": "tf.math.sqrt_1", "trainable": true, "dtype": "float32", "function": "math.sqrt"}, "inbound_nodes": [["tf.math.reduce_sum_1", 0, 0, {}]], "shared_object_id": 35}
о
М	keras_api"Ы
_tf_keras_layerБ{"name": "tf.math.sqrt_2", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": true, "class_name": "TFOpLambda", "config": {"name": "tf.math.sqrt_2", "trainable": true, "dtype": "float32", "function": "math.sqrt"}, "inbound_nodes": [["tf.math.reduce_sum_2", 0, 0, {}]], "shared_object_id": 36}
о
Н	keras_api"Ы
_tf_keras_layerБ{"name": "tf.math.sqrt_3", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": true, "class_name": "TFOpLambda", "config": {"name": "tf.math.sqrt_3", "trainable": true, "dtype": "float32", "function": "math.sqrt"}, "inbound_nodes": [["tf.math.reduce_sum_3", 0, 0, {}]], "shared_object_id": 37}
Ё
О	keras_api"▌
_tf_keras_layer├{"name": "tf.__operators__.add", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": true, "class_name": "TFOpLambda", "config": {"name": "tf.__operators__.add", "trainable": true, "dtype": "float32", "function": "__operators__.add"}, "inbound_nodes": [["decoder1/layer_gene", 0, 0, {"y": ["encoder1/fcn3_gene", 0, 0], "name": null}]], "shared_object_id": 38}
Ї
П	keras_api"с
_tf_keras_layer╟{"name": "tf.__operators__.add_2", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": true, "class_name": "TFOpLambda", "config": {"name": "tf.__operators__.add_2", "trainable": true, "dtype": "float32", "function": "__operators__.add"}, "inbound_nodes": [["decoder1/layer_gene", 1, 0, {"y": ["encoder1/fcn3_gene", 1, 0], "name": null}]], "shared_object_id": 39}
╘
Р	keras_api"┴
_tf_keras_layerз{"name": "tf.linalg.matmul", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": true, "class_name": "TFOpLambda", "config": {"name": "tf.linalg.matmul", "trainable": true, "dtype": "float32", "function": "linalg.matmul"}, "inbound_nodes": [["input_2", 0, 0, {"b": ["input_2", 0, 0], "transpose_b": true}]], "shared_object_id": 40}
▌
С	keras_api"╩
_tf_keras_layer░{"name": "tf.math.multiply_2", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": true, "class_name": "TFOpLambda", "config": {"name": "tf.math.multiply_2", "trainable": true, "dtype": "float32", "function": "math.multiply"}, "inbound_nodes": [["tf.math.sqrt", 0, 0, {"y": ["tf.math.sqrt_1", 0, 0], "name": null}]], "shared_object_id": 41}
ю
Т	keras_api"█
_tf_keras_layer┴{"name": "tf.linalg.matmul_1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": true, "class_name": "TFOpLambda", "config": {"name": "tf.linalg.matmul_1", "trainable": true, "dtype": "float32", "function": "linalg.matmul"}, "inbound_nodes": [["encoder2/fcn2/mean", 0, 0, {"b": ["encoder2/fcn2/mean", 0, 0], "transpose_b": true}]], "shared_object_id": 42}
▀
У	keras_api"╠
_tf_keras_layer▓{"name": "tf.math.multiply_3", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": true, "class_name": "TFOpLambda", "config": {"name": "tf.math.multiply_3", "trainable": true, "dtype": "float32", "function": "math.multiply"}, "inbound_nodes": [["tf.math.sqrt_2", 0, 0, {"y": ["tf.math.sqrt_3", 0, 0], "name": null}]], "shared_object_id": 43}
╨	
Фkernel
	Хbias
Ц	variables
Чregularization_losses
Шtrainable_variables
Щ	keras_api
+╕&call_and_return_all_conditional_losses
╣__call__"г
_tf_keras_layerЙ{"name": "decoder1/fcn1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Dense", "config": {"name": "decoder1/fcn1", "trainable": true, "dtype": "float32", "units": 1152, "activation": "leaky_relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 44}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 45}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "inbound_nodes": [[["tf.__operators__.add", 0, 0, {}]], [["tf.__operators__.add_2", 0, 0, {}]]], "shared_object_id": 46, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 100}}, "shared_object_id": 123}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 100]}}
╚	
Ъkernel
	Ыbias
Ь	variables
Эregularization_losses
Юtrainable_variables
Я	keras_api
+║&call_and_return_all_conditional_losses
╗__call__"Ы
_tf_keras_layerБ{"name": "decoder2/fcn1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Dense", "config": {"name": "decoder2/fcn1", "trainable": true, "dtype": "float32", "units": 1152, "activation": "leaky_relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 47}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 48}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "inbound_nodes": [[["encoder2/fcn2/mean", 0, 0, {}]], [["encoder2/fcn2/mean", 1, 0, {}]]], "shared_object_id": 49, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 96}}, "shared_object_id": 124}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 96]}}
▐
а	keras_api"╦
_tf_keras_layer▒{"name": "tf.math.truediv", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": true, "class_name": "TFOpLambda", "config": {"name": "tf.math.truediv", "trainable": true, "dtype": "float32", "function": "math.truediv"}, "inbound_nodes": [["tf.linalg.matmul", 0, 0, {"y": ["tf.math.multiply_2", 0, 0], "name": null}]], "shared_object_id": 50}
ф
б	keras_api"╤
_tf_keras_layer╖{"name": "tf.math.truediv_1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": true, "class_name": "TFOpLambda", "config": {"name": "tf.math.truediv_1", "trainable": true, "dtype": "float32", "function": "math.truediv"}, "inbound_nodes": [["tf.linalg.matmul_1", 0, 0, {"y": ["tf.math.multiply_3", 0, 0], "name": null}]], "shared_object_id": 51}
╜	
вkernel
	гbias
д	variables
еregularization_losses
жtrainable_variables
з	keras_api
+╝&call_and_return_all_conditional_losses
╜__call__"Р
_tf_keras_layerЎ{"name": "decoder1/fcn2", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Dense", "config": {"name": "decoder1/fcn2", "trainable": true, "dtype": "float32", "units": 768, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 52}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 53}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "inbound_nodes": [[["decoder1/fcn1", 0, 0, {}]], [["decoder1/fcn1", 1, 0, {}]]], "shared_object_id": 54, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 1152}}, "shared_object_id": 125}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 1152]}}
╜	
иkernel
	йbias
к	variables
лregularization_losses
мtrainable_variables
н	keras_api
+╛&call_and_return_all_conditional_losses
┐__call__"Р
_tf_keras_layerЎ{"name": "decoder2/fcn2", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Dense", "config": {"name": "decoder2/fcn2", "trainable": true, "dtype": "float32", "units": 768, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 55}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 56}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "inbound_nodes": [[["decoder2/fcn1", 0, 0, {}]], [["decoder2/fcn1", 1, 0, {}]]], "shared_object_id": 57, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 1152}}, "shared_object_id": 126}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 1152]}}
─
о	keras_api"▒
_tf_keras_layerЧ{"name": "tf.math.multiply_8", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": true, "class_name": "TFOpLambda", "config": {"name": "tf.math.multiply_8", "trainable": true, "dtype": "float32", "function": "math.multiply"}, "inbound_nodes": [["input_3", 0, 0, {"y": 50.0, "name": null}]], "shared_object_id": 58}
─
п	keras_api"▒
_tf_keras_layerЧ{"name": "tf.math.multiply_9", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": true, "class_name": "TFOpLambda", "config": {"name": "tf.math.multiply_9", "trainable": true, "dtype": "float32", "function": "math.multiply"}, "inbound_nodes": [["input_5", 0, 0, {"y": 50.0, "name": null}]], "shared_object_id": 59}
▒
░	keras_api"Ю
_tf_keras_layerД{"name": "tf.nn.softmax", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": true, "class_name": "TFOpLambda", "config": {"name": "tf.nn.softmax", "trainable": true, "dtype": "float32", "function": "nn.softmax"}, "inbound_nodes": [["tf.math.truediv", 0, 0, {"axis": 1}]], "shared_object_id": 60}
╖
▒	keras_api"д
_tf_keras_layerК{"name": "tf.nn.softmax_1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": true, "class_name": "TFOpLambda", "config": {"name": "tf.nn.softmax_1", "trainable": true, "dtype": "float32", "function": "nn.softmax"}, "inbound_nodes": [["tf.math.truediv_1", 0, 0, {"axis": 1}]], "shared_object_id": 61}
└
▓	keras_api"н
_tf_keras_layerУ{"name": "tf.math.multiply", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": true, "class_name": "TFOpLambda", "config": {"name": "tf.math.multiply", "trainable": true, "dtype": "float32", "function": "math.multiply"}, "inbound_nodes": [["input_1", 0, 0, {"y": 50.0, "name": null}]], "shared_object_id": 62}
─
│	keras_api"▒
_tf_keras_layerЧ{"name": "tf.math.multiply_1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": true, "class_name": "TFOpLambda", "config": {"name": "tf.math.multiply_1", "trainable": true, "dtype": "float32", "function": "math.multiply"}, "inbound_nodes": [["input_2", 0, 0, {"y": 50.0, "name": null}]], "shared_object_id": 63}
т
┤	keras_api"╧
_tf_keras_layer╡{"name": "tf.math.subtract_4", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": true, "class_name": "TFOpLambda", "config": {"name": "tf.math.subtract_4", "trainable": true, "dtype": "float32", "function": "math.subtract"}, "inbound_nodes": [["tf.math.multiply_8", 0, 0, {"y": ["decoder1/fcn2", 1, 0], "name": null}]], "shared_object_id": 64}
т
╡	keras_api"╧
_tf_keras_layer╡{"name": "tf.math.subtract_5", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": true, "class_name": "TFOpLambda", "config": {"name": "tf.math.subtract_5", "trainable": true, "dtype": "float32", "function": "math.subtract"}, "inbound_nodes": [["tf.math.multiply_9", 0, 0, {"y": ["decoder2/fcn2", 1, 0], "name": null}]], "shared_object_id": 65}
▄
╢	keras_api"╔
_tf_keras_layerп{"name": "tf.math.truediv_2", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": true, "class_name": "TFOpLambda", "config": {"name": "tf.math.truediv_2", "trainable": true, "dtype": "float32", "function": "math.truediv"}, "inbound_nodes": [["tf.nn.softmax", 0, 0, {"y": ["tf.nn.softmax_1", 0, 0], "name": null}]], "shared_object_id": 66}
▄
╖	keras_api"╔
_tf_keras_layerп{"name": "tf.math.truediv_3", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": true, "class_name": "TFOpLambda", "config": {"name": "tf.math.truediv_3", "trainable": true, "dtype": "float32", "function": "math.truediv"}, "inbound_nodes": [["tf.nn.softmax_1", 0, 0, {"y": ["tf.nn.softmax", 0, 0], "name": null}]], "shared_object_id": 67}
р
╕	keras_api"═
_tf_keras_layer│{"name": "tf.math.subtract_1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": true, "class_name": "TFOpLambda", "config": {"name": "tf.math.subtract_1", "trainable": true, "dtype": "float32", "function": "math.subtract"}, "inbound_nodes": [["tf.math.multiply", 0, 0, {"y": ["decoder1/fcn2", 0, 0], "name": null}]], "shared_object_id": 68}
т
╣	keras_api"╧
_tf_keras_layer╡{"name": "tf.math.subtract_2", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": true, "class_name": "TFOpLambda", "config": {"name": "tf.math.subtract_2", "trainable": true, "dtype": "float32", "function": "math.subtract"}, "inbound_nodes": [["tf.math.multiply_1", 0, 0, {"y": ["decoder2/fcn2", 0, 0], "name": null}]], "shared_object_id": 69}
╛
║	keras_api"л
_tf_keras_layerС{"name": "tf.math.square_8", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": true, "class_name": "TFOpLambda", "config": {"name": "tf.math.square_8", "trainable": true, "dtype": "float32", "function": "math.square"}, "inbound_nodes": [["tf.math.subtract_4", 0, 0, {"name": null}]], "shared_object_id": 70}
╛
╗	keras_api"л
_tf_keras_layerС{"name": "tf.math.square_9", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": true, "class_name": "TFOpLambda", "config": {"name": "tf.math.square_9", "trainable": true, "dtype": "float32", "function": "math.square"}, "inbound_nodes": [["tf.math.subtract_5", 0, 0, {"name": null}]], "shared_object_id": 71}
░
╝	keras_api"Э
_tf_keras_layerГ{"name": "tf.math.log", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": true, "class_name": "TFOpLambda", "config": {"name": "tf.math.log", "trainable": true, "dtype": "float32", "function": "math.log"}, "inbound_nodes": [["tf.math.truediv_2", 0, 0, {"name": null}]], "shared_object_id": 72}
┤
╜	keras_api"б
_tf_keras_layerЗ{"name": "tf.math.log_1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": true, "class_name": "TFOpLambda", "config": {"name": "tf.math.log_1", "trainable": true, "dtype": "float32", "function": "math.log"}, "inbound_nodes": [["tf.math.truediv_3", 0, 0, {"name": null}]], "shared_object_id": 73}
а	
╛kernel
	┐bias
└	variables
┴regularization_losses
┬trainable_variables
├	keras_api
+└&call_and_return_all_conditional_losses
┴__call__"є
_tf_keras_layer┘{"name": "encoder1/fcn3", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Dense", "config": {"name": "encoder1/fcn3", "trainable": true, "dtype": "float32", "units": 80, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 74}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 75}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "inbound_nodes": [[["encoder1/fcn2/mean", 0, 0, {}]]], "shared_object_id": 76, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 100}}, "shared_object_id": 127}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 100]}}
Ю	
─kernel
	┼bias
╞	variables
╟regularization_losses
╚trainable_variables
╔	keras_api
+┬&call_and_return_all_conditional_losses
├__call__"ё
_tf_keras_layer╫{"name": "encoder2/fcn3", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Dense", "config": {"name": "encoder2/fcn3", "trainable": true, "dtype": "float32", "units": 80, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 77}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 78}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "inbound_nodes": [[["encoder2/fcn2/mean", 0, 0, {}]]], "shared_object_id": 79, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 96}}, "shared_object_id": 128}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 96]}}
║
╩	keras_api"з
_tf_keras_layerН{"name": "tf.math.square", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": true, "class_name": "TFOpLambda", "config": {"name": "tf.math.square", "trainable": true, "dtype": "float32", "function": "math.square"}, "inbound_nodes": [["tf.math.subtract_1", 0, 0, {"name": null}]], "shared_object_id": 80}
╛
╦	keras_api"л
_tf_keras_layerС{"name": "tf.math.square_1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": true, "class_name": "TFOpLambda", "config": {"name": "tf.math.square_1", "trainable": true, "dtype": "float32", "function": "math.square"}, "inbound_nodes": [["tf.math.subtract_2", 0, 0, {"name": null}]], "shared_object_id": 81}
┐
╠	keras_api"м
_tf_keras_layerТ{"name": "tf.math.reduce_mean_3", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": true, "class_name": "TFOpLambda", "config": {"name": "tf.math.reduce_mean_3", "trainable": true, "dtype": "float32", "function": "math.reduce_mean"}, "inbound_nodes": [["tf.math.square_8", 0, 0, {}]], "shared_object_id": 82}
┐
═	keras_api"м
_tf_keras_layerТ{"name": "tf.math.reduce_mean_4", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": true, "class_name": "TFOpLambda", "config": {"name": "tf.math.reduce_mean_4", "trainable": true, "dtype": "float32", "function": "math.reduce_mean"}, "inbound_nodes": [["tf.math.square_9", 0, 0, {}]], "shared_object_id": 83}
█
╬	keras_api"╚
_tf_keras_layerо{"name": "tf.math.multiply_4", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": true, "class_name": "TFOpLambda", "config": {"name": "tf.math.multiply_4", "trainable": true, "dtype": "float32", "function": "math.multiply"}, "inbound_nodes": [["tf.nn.softmax", 0, 0, {"y": ["tf.math.log", 0, 0], "name": null}]], "shared_object_id": 84}
▀
╧	keras_api"╠
_tf_keras_layer▓{"name": "tf.math.multiply_5", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": true, "class_name": "TFOpLambda", "config": {"name": "tf.math.multiply_5", "trainable": true, "dtype": "float32", "function": "math.multiply"}, "inbound_nodes": [["tf.nn.softmax_1", 0, 0, {"y": ["tf.math.log_1", 0, 0], "name": null}]], "shared_object_id": 85}
╣
╨	keras_api"ж
_tf_keras_layerМ{"name": "tf.math.square_6", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": true, "class_name": "TFOpLambda", "config": {"name": "tf.math.square_6", "trainable": true, "dtype": "float32", "function": "math.square"}, "inbound_nodes": [["encoder1/fcn3", 0, 0, {"name": null}]], "shared_object_id": 86}
╣
╤	keras_api"ж
_tf_keras_layerМ{"name": "tf.math.square_7", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": true, "class_name": "TFOpLambda", "config": {"name": "tf.math.square_7", "trainable": true, "dtype": "float32", "function": "math.square"}, "inbound_nodes": [["encoder2/fcn3", 0, 0, {"name": null}]], "shared_object_id": 87}
╣
╥	keras_api"ж
_tf_keras_layerМ{"name": "tf.math.reduce_mean", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": true, "class_name": "TFOpLambda", "config": {"name": "tf.math.reduce_mean", "trainable": true, "dtype": "float32", "function": "math.reduce_mean"}, "inbound_nodes": [["tf.math.square", 0, 0, {}]], "shared_object_id": 88}
┐
╙	keras_api"м
_tf_keras_layerТ{"name": "tf.math.reduce_mean_1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": true, "class_name": "TFOpLambda", "config": {"name": "tf.math.reduce_mean_1", "trainable": true, "dtype": "float32", "function": "math.reduce_mean"}, "inbound_nodes": [["tf.math.square_1", 0, 0, {}]], "shared_object_id": 89}
√
╘	keras_api"ш
_tf_keras_layer╬{"name": "tf.__operators__.add_16", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": true, "class_name": "TFOpLambda", "config": {"name": "tf.__operators__.add_16", "trainable": true, "dtype": "float32", "function": "__operators__.add"}, "inbound_nodes": [["tf.math.reduce_mean_3", 0, 0, {"y": ["tf.math.reduce_mean_4", 0, 0], "name": null}]], "shared_object_id": 90}
╚
╒	keras_api"╡
_tf_keras_layerЫ{"name": "tf.math.reduce_sum_4", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": true, "class_name": "TFOpLambda", "config": {"name": "tf.math.reduce_sum_4", "trainable": true, "dtype": "float32", "function": "math.reduce_sum"}, "inbound_nodes": [["tf.math.multiply_4", 0, 0, {"axis": -1}]], "shared_object_id": 91}
╚
╓	keras_api"╡
_tf_keras_layerЫ{"name": "tf.math.reduce_sum_5", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": true, "class_name": "TFOpLambda", "config": {"name": "tf.math.reduce_sum_5", "trainable": true, "dtype": "float32", "function": "math.reduce_sum"}, "inbound_nodes": [["tf.math.multiply_5", 0, 0, {"axis": -1}]], "shared_object_id": 92}
╞
╫	keras_api"│
_tf_keras_layerЩ{"name": "tf.math.reduce_sum_7", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": true, "class_name": "TFOpLambda", "config": {"name": "tf.math.reduce_sum_7", "trainable": true, "dtype": "float32", "function": "math.reduce_sum"}, "inbound_nodes": [["tf.math.square_6", 0, 0, {"axis": -1}]], "shared_object_id": 93}
╞
╪	keras_api"│
_tf_keras_layerЩ{"name": "tf.math.reduce_sum_8", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": true, "class_name": "TFOpLambda", "config": {"name": "tf.math.reduce_sum_8", "trainable": true, "dtype": "float32", "function": "math.reduce_sum"}, "inbound_nodes": [["tf.math.square_7", 0, 0, {"axis": -1}]], "shared_object_id": 94}
∙
┘	keras_api"ц
_tf_keras_layer╠{"name": "tf.__operators__.add_15", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": true, "class_name": "TFOpLambda", "config": {"name": "tf.__operators__.add_15", "trainable": true, "dtype": "float32", "function": "__operators__.add"}, "inbound_nodes": [["tf.math.reduce_mean", 0, 0, {"y": ["tf.math.reduce_mean_1", 0, 0], "name": null}]], "shared_object_id": 95}
■
┌	keras_api"ы
_tf_keras_layer╤{"name": "tf.math.multiply_22", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": true, "class_name": "TFOpLambda", "config": {"name": "tf.math.multiply_22", "trainable": true, "dtype": "float32", "function": "math.multiply"}, "inbound_nodes": [["_CONSTANT_VALUE", -1, 0.20000000298023224, {"y": ["tf.__operators__.add_16", 0, 0], "name": null}]], "shared_object_id": 96}
ў
█	keras_api"ф
_tf_keras_layer╩{"name": "tf.__operators__.add_1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": true, "class_name": "TFOpLambda", "config": {"name": "tf.__operators__.add_1", "trainable": true, "dtype": "float32", "function": "__operators__.add"}, "inbound_nodes": [["tf.math.reduce_sum_4", 0, 0, {"y": ["tf.math.reduce_sum_5", 0, 0], "name": null}]], "shared_object_id": 97}
╧
▄	keras_api"╝
_tf_keras_layerв{"name": "tf.math.multiply_6", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": true, "class_name": "TFOpLambda", "config": {"name": "tf.math.multiply_6", "trainable": true, "dtype": "float32", "function": "math.multiply"}, "inbound_nodes": [["encoder1/fcn3", 0, 0, {"y": ["encoder2/fcn3", 0, 0]}]], "shared_object_id": 98}
о
▌	keras_api"Ы
_tf_keras_layerБ{"name": "tf.math.sqrt_4", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": true, "class_name": "TFOpLambda", "config": {"name": "tf.math.sqrt_4", "trainable": true, "dtype": "float32", "function": "math.sqrt"}, "inbound_nodes": [["tf.math.reduce_sum_7", 0, 0, {}]], "shared_object_id": 99}
п
▐	keras_api"Ь
_tf_keras_layerВ{"name": "tf.math.sqrt_5", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": true, "class_name": "TFOpLambda", "config": {"name": "tf.math.sqrt_5", "trainable": true, "dtype": "float32", "function": "math.sqrt"}, "inbound_nodes": [["tf.math.reduce_sum_8", 0, 0, {}]], "shared_object_id": 100}
╙
▀	keras_api"└
_tf_keras_layerж{"name": "tf.math.truediv_21", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": true, "class_name": "TFOpLambda", "config": {"name": "tf.math.truediv_21", "trainable": true, "dtype": "float32", "function": "math.truediv"}, "inbound_nodes": [["tf.__operators__.add_15", 0, 0, {"y": 2.5, "name": null}]], "shared_object_id": 101}
╧
р	keras_api"╝
_tf_keras_layerв{"name": "tf.math.truediv_22", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": true, "class_name": "TFOpLambda", "config": {"name": "tf.math.truediv_22", "trainable": true, "dtype": "float32", "function": "math.truediv"}, "inbound_nodes": [["tf.math.multiply_22", 0, 0, {"y": 2.0, "name": null}]], "shared_object_id": 102}
╞
с	keras_api"│
_tf_keras_layerЩ{"name": "tf.math.reduce_mean_2", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": true, "class_name": "TFOpLambda", "config": {"name": "tf.math.reduce_mean_2", "trainable": true, "dtype": "float32", "function": "math.reduce_mean"}, "inbound_nodes": [["tf.__operators__.add_1", 0, 0, {}]], "shared_object_id": 103}
╔
т	keras_api"╢
_tf_keras_layerЬ{"name": "tf.math.reduce_sum_6", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": true, "class_name": "TFOpLambda", "config": {"name": "tf.math.reduce_sum_6", "trainable": true, "dtype": "float32", "function": "math.reduce_sum"}, "inbound_nodes": [["tf.math.multiply_6", 0, 0, {"axis": -1}]], "shared_object_id": 104}
р
у	keras_api"═
_tf_keras_layer│{"name": "tf.math.multiply_7", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": true, "class_name": "TFOpLambda", "config": {"name": "tf.math.multiply_7", "trainable": true, "dtype": "float32", "function": "math.multiply"}, "inbound_nodes": [["tf.math.sqrt_4", 0, 0, {"y": ["tf.math.sqrt_5", 0, 0], "name": null}]], "shared_object_id": 105}
Ў
ф	keras_api"у
_tf_keras_layer╔{"name": "tf.__operators__.add_17", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": true, "class_name": "TFOpLambda", "config": {"name": "tf.__operators__.add_17", "trainable": true, "dtype": "float32", "function": "__operators__.add"}, "inbound_nodes": [["tf.math.truediv_21", 0, 0, {"y": ["tf.math.truediv_22", 0, 0], "name": null}]], "shared_object_id": 106}
╘
х	keras_api"┴
_tf_keras_layerз{"name": "tf.math.multiply_23", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": true, "class_name": "TFOpLambda", "config": {"name": "tf.math.multiply_23", "trainable": true, "dtype": "float32", "function": "math.multiply"}, "inbound_nodes": [["tf.math.reduce_mean_2", 0, 0, {"y": 0.5, "name": null}]], "shared_object_id": 107}
ч
ц	keras_api"╘
_tf_keras_layer║{"name": "tf.math.truediv_4", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": true, "class_name": "TFOpLambda", "config": {"name": "tf.math.truediv_4", "trainable": true, "dtype": "float32", "function": "math.truediv"}, "inbound_nodes": [["tf.math.reduce_sum_6", 0, 0, {"y": ["tf.math.multiply_7", 0, 0], "name": null}]], "shared_object_id": 108}
№
ч	keras_api"щ
_tf_keras_layer╧{"name": "tf.__operators__.add_18", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": true, "class_name": "TFOpLambda", "config": {"name": "tf.__operators__.add_18", "trainable": true, "dtype": "float32", "function": "__operators__.add"}, "inbound_nodes": [["tf.__operators__.add_17", 0, 0, {"y": ["tf.math.multiply_23", 0, 0], "name": null}]], "shared_object_id": 109}
ф
\0
]1
b2
c3
h4
i5
n6
o7
z8
{9
Д10
Е11
Ф12
Х13
Ъ14
Ы15
в16
г17
и18
й19
╛20
┐21
─22
┼23"
trackable_list_wrapper
 "
trackable_list_wrapper
ф
\0
]1
b2
c3
h4
i5
n6
o7
z8
{9
Д10
Е11
Ф12
Х13
Ъ14
Ы15
в16
г17
и18
й19
╛20
┐21
─22
┼23"
trackable_list_wrapper
╙
шmetrics
W	variables
Xregularization_losses
щlayers
ъlayer_metrics
Ytrainable_variables
ыnon_trainable_variables
 ьlayer_regularization_losses
л__call__
й_default_save_signature
+к&call_and_return_all_conditional_losses
'к"call_and_return_conditional_losses"
_generic_user_object
-
─serving_default"
signature_map
':%	Аx2encoder2/fcn1/kernel
 :x2encoder2/fcn1/bias
.
\0
]1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
\0
]1"
trackable_list_wrapper
╡
эlayer_metrics
юmetrics
^	variables
яlayers
_regularization_losses
 Ёlayer_regularization_losses
`trainable_variables
ёnon_trainable_variables
н__call__
+м&call_and_return_all_conditional_losses
'м"call_and_return_conditional_losses"
_generic_user_object
':%	Аd2encoder1/fcn1/kernel
 :d2encoder1/fcn1/bias
.
b0
c1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
b0
c1"
trackable_list_wrapper
╡
Єlayer_metrics
єmetrics
d	variables
Їlayers
eregularization_losses
 їlayer_regularization_losses
ftrainable_variables
Ўnon_trainable_variables
п__call__
+о&call_and_return_all_conditional_losses
'о"call_and_return_conditional_losses"
_generic_user_object
,:*	Аd2encoder1/fcn3_gene/kernel
%:#d2encoder1/fcn3_gene/bias
.
h0
i1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
h0
i1"
trackable_list_wrapper
╡
ўlayer_metrics
°metrics
j	variables
∙layers
kregularization_losses
 ·layer_regularization_losses
ltrainable_variables
√non_trainable_variables
▒__call__
+░&call_and_return_all_conditional_losses
'░"call_and_return_conditional_losses"
_generic_user_object
+:)x`2encoder2/fcn2/mean/kernel
%:#`2encoder2/fcn2/mean/bias
.
n0
o1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
n0
o1"
trackable_list_wrapper
╡
№layer_metrics
¤metrics
p	variables
■layers
qregularization_losses
  layer_regularization_losses
rtrainable_variables
Аnon_trainable_variables
│__call__
+▓&call_and_return_all_conditional_losses
'▓"call_and_return_conditional_losses"
_generic_user_object
"
_generic_user_object
"
_generic_user_object
"
_generic_user_object
"
_generic_user_object
"
_generic_user_object
"
_generic_user_object
+:)dd2encoder1/fcn2/mean/kernel
%:#d2encoder1/fcn2/mean/bias
.
z0
{1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
z0
{1"
trackable_list_wrapper
╡
Бlayer_metrics
Вmetrics
|	variables
Гlayers
}regularization_losses
 Дlayer_regularization_losses
~trainable_variables
Еnon_trainable_variables
╡__call__
+┤&call_and_return_all_conditional_losses
'┤"call_and_return_conditional_losses"
_generic_user_object
"
_generic_user_object
"
_generic_user_object
"
_generic_user_object
"
_generic_user_object
,:*dd2decoder1/layer_gene/kernel
&:$d2decoder1/layer_gene/bias
0
Д0
Е1"
trackable_list_wrapper
 "
trackable_list_wrapper
0
Д0
Е1"
trackable_list_wrapper
╕
Жlayer_metrics
Зmetrics
Ж	variables
Иlayers
Зregularization_losses
 Йlayer_regularization_losses
Иtrainable_variables
Кnon_trainable_variables
╖__call__
+╢&call_and_return_all_conditional_losses
'╢"call_and_return_conditional_losses"
_generic_user_object
"
_generic_user_object
"
_generic_user_object
"
_generic_user_object
"
_generic_user_object
"
_generic_user_object
"
_generic_user_object
"
_generic_user_object
"
_generic_user_object
"
_generic_user_object
"
_generic_user_object
':%	dА	2decoder1/fcn1/kernel
!:А	2decoder1/fcn1/bias
0
Ф0
Х1"
trackable_list_wrapper
 "
trackable_list_wrapper
0
Ф0
Х1"
trackable_list_wrapper
╕
Лlayer_metrics
Мmetrics
Ц	variables
Нlayers
Чregularization_losses
 Оlayer_regularization_losses
Шtrainable_variables
Пnon_trainable_variables
╣__call__
+╕&call_and_return_all_conditional_losses
'╕"call_and_return_conditional_losses"
_generic_user_object
':%	`А	2decoder2/fcn1/kernel
!:А	2decoder2/fcn1/bias
0
Ъ0
Ы1"
trackable_list_wrapper
 "
trackable_list_wrapper
0
Ъ0
Ы1"
trackable_list_wrapper
╕
Рlayer_metrics
Сmetrics
Ь	variables
Тlayers
Эregularization_losses
 Уlayer_regularization_losses
Юtrainable_variables
Фnon_trainable_variables
╗__call__
+║&call_and_return_all_conditional_losses
'║"call_and_return_conditional_losses"
_generic_user_object
"
_generic_user_object
"
_generic_user_object
(:&
А	А2decoder1/fcn2/kernel
!:А2decoder1/fcn2/bias
0
в0
г1"
trackable_list_wrapper
 "
trackable_list_wrapper
0
в0
г1"
trackable_list_wrapper
╕
Хlayer_metrics
Цmetrics
д	variables
Чlayers
еregularization_losses
 Шlayer_regularization_losses
жtrainable_variables
Щnon_trainable_variables
╜__call__
+╝&call_and_return_all_conditional_losses
'╝"call_and_return_conditional_losses"
_generic_user_object
(:&
А	А2decoder2/fcn2/kernel
!:А2decoder2/fcn2/bias
0
и0
й1"
trackable_list_wrapper
 "
trackable_list_wrapper
0
и0
й1"
trackable_list_wrapper
╕
Ъlayer_metrics
Ыmetrics
к	variables
Ьlayers
лregularization_losses
 Эlayer_regularization_losses
мtrainable_variables
Юnon_trainable_variables
┐__call__
+╛&call_and_return_all_conditional_losses
'╛"call_and_return_conditional_losses"
_generic_user_object
"
_generic_user_object
"
_generic_user_object
"
_generic_user_object
"
_generic_user_object
"
_generic_user_object
"
_generic_user_object
"
_generic_user_object
"
_generic_user_object
"
_generic_user_object
"
_generic_user_object
"
_generic_user_object
"
_generic_user_object
"
_generic_user_object
"
_generic_user_object
"
_generic_user_object
"
_generic_user_object
&:$dP2encoder1/fcn3/kernel
 :P2encoder1/fcn3/bias
0
╛0
┐1"
trackable_list_wrapper
 "
trackable_list_wrapper
0
╛0
┐1"
trackable_list_wrapper
╕
Яlayer_metrics
аmetrics
└	variables
бlayers
┴regularization_losses
 вlayer_regularization_losses
┬trainable_variables
гnon_trainable_variables
┴__call__
+└&call_and_return_all_conditional_losses
'└"call_and_return_conditional_losses"
_generic_user_object
&:$`P2encoder2/fcn3/kernel
 :P2encoder2/fcn3/bias
0
─0
┼1"
trackable_list_wrapper
 "
trackable_list_wrapper
0
─0
┼1"
trackable_list_wrapper
╕
дlayer_metrics
еmetrics
╞	variables
жlayers
╟regularization_losses
 зlayer_regularization_losses
╚trainable_variables
иnon_trainable_variables
├__call__
+┬&call_and_return_all_conditional_losses
'┬"call_and_return_conditional_losses"
_generic_user_object
"
_generic_user_object
"
_generic_user_object
"
_generic_user_object
"
_generic_user_object
"
_generic_user_object
"
_generic_user_object
"
_generic_user_object
"
_generic_user_object
"
_generic_user_object
"
_generic_user_object
"
_generic_user_object
"
_generic_user_object
"
_generic_user_object
"
_generic_user_object
"
_generic_user_object
"
_generic_user_object
"
_generic_user_object
"
_generic_user_object
"
_generic_user_object
"
_generic_user_object
"
_generic_user_object
"
_generic_user_object
"
_generic_user_object
"
_generic_user_object
"
_generic_user_object
"
_generic_user_object
"
_generic_user_object
"
_generic_user_object
"
_generic_user_object
"
_generic_user_object
 "
trackable_list_wrapper
╞
0
1
2
3
4
5
6
7
	8

9
10
11
12
13
14
15
16
17
18
19
20
21
22
23
24
25
26
27
28
29
30
 31
!32
"33
#34
$35
%36
&37
'38
(39
)40
*41
+42
,43
-44
.45
/46
047
148
249
350
451
552
653
754
855
956
:57
;58
<59
=60
>61
?62
@63
A64
B65
C66
D67
E68
F69
G70
H71
I72
J73
K74
L75
M76
N77
O78
P79
Q80
R81
S82
T83
U84
V85"
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
Я2Ь
#__inference__wrapped_model_40049440Ї
Л▓З
FullArgSpec
argsЪ 
varargsjargs
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *ув▀
▄Ъ╪
"К
input_1         А
"К
input_2         А
"К
input_3         А
"К
input_4         А
"К
input_5         А
"К
input_6         А
┌2╫
C__inference_model_layer_call_and_return_conditional_losses_40051034
C__inference_model_layer_call_and_return_conditional_losses_40051263
C__inference_model_layer_call_and_return_conditional_losses_40050544
C__inference_model_layer_call_and_return_conditional_losses_40050737└
╖▓│
FullArgSpec1
args)Ъ&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsЪ
p 

 

kwonlyargsЪ 
kwonlydefaultsк 
annotationsк *
 
ю2ы
(__inference_model_layer_call_fn_40049840
(__inference_model_layer_call_fn_40051329
(__inference_model_layer_call_fn_40051395
(__inference_model_layer_call_fn_40050351└
╖▓│
FullArgSpec1
args)Ъ&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsЪ
p 

 

kwonlyargsЪ 
kwonlydefaultsк 
annotationsк *
 
ї2Є
K__inference_encoder2/fcn1_layer_call_and_return_conditional_losses_40051406в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
┌2╫
0__inference_encoder2/fcn1_layer_call_fn_40051415в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
ї2Є
K__inference_encoder1/fcn1_layer_call_and_return_conditional_losses_40051426в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
┌2╫
0__inference_encoder1/fcn1_layer_call_fn_40051435в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
·2ў
P__inference_encoder1/fcn3_gene_layer_call_and_return_conditional_losses_40051445в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
▀2▄
5__inference_encoder1/fcn3_gene_layer_call_fn_40051454в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
·2ў
P__inference_encoder2/fcn2/mean_layer_call_and_return_conditional_losses_40051465в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
▀2▄
5__inference_encoder2/fcn2/mean_layer_call_fn_40051474в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
·2ў
P__inference_encoder1/fcn2/mean_layer_call_and_return_conditional_losses_40051485в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
▀2▄
5__inference_encoder1/fcn2/mean_layer_call_fn_40051494в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
√2°
Q__inference_decoder1/layer_gene_layer_call_and_return_conditional_losses_40051505в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
р2▌
6__inference_decoder1/layer_gene_layer_call_fn_40051514в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
ї2Є
K__inference_decoder1/fcn1_layer_call_and_return_conditional_losses_40051525в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
┌2╫
0__inference_decoder1/fcn1_layer_call_fn_40051534в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
ї2Є
K__inference_decoder2/fcn1_layer_call_and_return_conditional_losses_40051545в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
┌2╫
0__inference_decoder2/fcn1_layer_call_fn_40051554в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
ї2Є
K__inference_decoder1/fcn2_layer_call_and_return_conditional_losses_40051564в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
┌2╫
0__inference_decoder1/fcn2_layer_call_fn_40051573в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
ї2Є
K__inference_decoder2/fcn2_layer_call_and_return_conditional_losses_40051583в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
┌2╫
0__inference_decoder2/fcn2_layer_call_fn_40051592в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
ї2Є
K__inference_encoder1/fcn3_layer_call_and_return_conditional_losses_40051602в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
┌2╫
0__inference_encoder1/fcn3_layer_call_fn_40051611в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
ї2Є
K__inference_encoder2/fcn3_layer_call_and_return_conditional_losses_40051621в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
┌2╫
0__inference_encoder2/fcn3_layer_call_fn_40051630в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
°Bї
&__inference_signature_wrapper_40050805input_1input_2input_3input_4input_5input_6"Ф
Н▓Й
FullArgSpec
argsЪ 
varargs
 
varkwjkwargs
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
	J
Const║
#__inference__wrapped_model_40049440Т(\]nohibcz{ДЕЪЫФХийвг─┼╛┐┼явы
ув▀
▄Ъ╪
"К
input_1         А
"К
input_2         А
"К
input_3         А
"К
input_4         А
"К
input_5         А
"К
input_6         А
к "єкя
8
encoder1/fcn3'К$
encoder1/fcn3         P
8
encoder2/fcn3'К$
encoder2/fcn3         P
;
tf.__operators__.add_18 К
tf.__operators__.add_18 
<
tf.math.truediv_4'К$
tf.math.truediv_4         о
K__inference_decoder1/fcn1_layer_call_and_return_conditional_losses_40051525_ФХ/в,
%в"
 К
inputs         d
к "&в#
К
0         А	
Ъ Ж
0__inference_decoder1/fcn1_layer_call_fn_40051534RФХ/в,
%в"
 К
inputs         d
к "К         А	п
K__inference_decoder1/fcn2_layer_call_and_return_conditional_losses_40051564`вг0в-
&в#
!К
inputs         А	
к "&в#
К
0         А
Ъ З
0__inference_decoder1/fcn2_layer_call_fn_40051573Sвг0в-
&в#
!К
inputs         А	
к "К         А│
Q__inference_decoder1/layer_gene_layer_call_and_return_conditional_losses_40051505^ДЕ/в,
%в"
 К
inputs         d
к "%в"
К
0         d
Ъ Л
6__inference_decoder1/layer_gene_layer_call_fn_40051514QДЕ/в,
%в"
 К
inputs         d
к "К         dо
K__inference_decoder2/fcn1_layer_call_and_return_conditional_losses_40051545_ЪЫ/в,
%в"
 К
inputs         `
к "&в#
К
0         А	
Ъ Ж
0__inference_decoder2/fcn1_layer_call_fn_40051554RЪЫ/в,
%в"
 К
inputs         `
к "К         А	п
K__inference_decoder2/fcn2_layer_call_and_return_conditional_losses_40051583`ий0в-
&в#
!К
inputs         А	
к "&в#
К
0         А
Ъ З
0__inference_decoder2/fcn2_layer_call_fn_40051592Sий0в-
&в#
!К
inputs         А	
к "К         Ам
K__inference_encoder1/fcn1_layer_call_and_return_conditional_losses_40051426]bc0в-
&в#
!К
inputs         А
к "%в"
К
0         d
Ъ Д
0__inference_encoder1/fcn1_layer_call_fn_40051435Pbc0в-
&в#
!К
inputs         А
к "К         d░
P__inference_encoder1/fcn2/mean_layer_call_and_return_conditional_losses_40051485\z{/в,
%в"
 К
inputs         d
к "%в"
К
0         d
Ъ И
5__inference_encoder1/fcn2/mean_layer_call_fn_40051494Oz{/в,
%в"
 К
inputs         d
к "К         d▒
P__inference_encoder1/fcn3_gene_layer_call_and_return_conditional_losses_40051445]hi0в-
&в#
!К
inputs         А
к "%в"
К
0         d
Ъ Й
5__inference_encoder1/fcn3_gene_layer_call_fn_40051454Phi0в-
&в#
!К
inputs         А
к "К         dн
K__inference_encoder1/fcn3_layer_call_and_return_conditional_losses_40051602^╛┐/в,
%в"
 К
inputs         d
к "%в"
К
0         P
Ъ Е
0__inference_encoder1/fcn3_layer_call_fn_40051611Q╛┐/в,
%в"
 К
inputs         d
к "К         Pм
K__inference_encoder2/fcn1_layer_call_and_return_conditional_losses_40051406]\]0в-
&в#
!К
inputs         А
к "%в"
К
0         x
Ъ Д
0__inference_encoder2/fcn1_layer_call_fn_40051415P\]0в-
&в#
!К
inputs         А
к "К         x░
P__inference_encoder2/fcn2/mean_layer_call_and_return_conditional_losses_40051465\no/в,
%в"
 К
inputs         x
к "%в"
К
0         `
Ъ И
5__inference_encoder2/fcn2/mean_layer_call_fn_40051474Ono/в,
%в"
 К
inputs         x
к "К         `н
K__inference_encoder2/fcn3_layer_call_and_return_conditional_losses_40051621^─┼/в,
%в"
 К
inputs         `
к "%в"
К
0         P
Ъ Е
0__inference_encoder2/fcn3_layer_call_fn_40051630Q─┼/в,
%в"
 К
inputs         `
к "К         Pт
C__inference_model_layer_call_and_return_conditional_losses_40050544Ъ(\]nohibcz{ДЕЪЫФХийвг─┼╛┐┼ўвє
ывч
▄Ъ╪
"К
input_1         А
"К
input_2         А
"К
input_3         А
"К
input_4         А
"К
input_5         А
"К
input_6         А
p 

 
к "tвq
jЪg
К
0/0         
К	
0/1 
К
0/2         P
К
0/3         P
Ъ т
C__inference_model_layer_call_and_return_conditional_losses_40050737Ъ(\]nohibcz{ДЕЪЫФХийвг─┼╛┐┼ўвє
ывч
▄Ъ╪
"К
input_1         А
"К
input_2         А
"К
input_3         А
"К
input_4         А
"К
input_5         А
"К
input_6         А
p

 
к "tвq
jЪg
К
0/0         
К	
0/1 
К
0/2         P
К
0/3         P
Ъ ш
C__inference_model_layer_call_and_return_conditional_losses_40051034а(\]nohibcz{ДЕЪЫФХийвг─┼╛┐┼¤в∙
ёвэ
тЪ▐
#К 
inputs/0         А
#К 
inputs/1         А
#К 
inputs/2         А
#К 
inputs/3         А
#К 
inputs/4         А
#К 
inputs/5         А
p 

 
к "tвq
jЪg
К
0/0         
К	
0/1 
К
0/2         P
К
0/3         P
Ъ ш
C__inference_model_layer_call_and_return_conditional_losses_40051263а(\]nohibcz{ДЕЪЫФХийвг─┼╛┐┼¤в∙
ёвэ
тЪ▐
#К 
inputs/0         А
#К 
inputs/1         А
#К 
inputs/2         А
#К 
inputs/3         А
#К 
inputs/4         А
#К 
inputs/5         А
p

 
к "tвq
jЪg
К
0/0         
К	
0/1 
К
0/2         P
К
0/3         P
Ъ ╡
(__inference_model_layer_call_fn_40049840И(\]nohibcz{ДЕЪЫФХийвг─┼╛┐┼ўвє
ывч
▄Ъ╪
"К
input_1         А
"К
input_2         А
"К
input_3         А
"К
input_4         А
"К
input_5         А
"К
input_6         А
p 

 
к "bЪ_
К
0         

К
1 
К
2         P
К
3         P╡
(__inference_model_layer_call_fn_40050351И(\]nohibcz{ДЕЪЫФХийвг─┼╛┐┼ўвє
ывч
▄Ъ╪
"К
input_1         А
"К
input_2         А
"К
input_3         А
"К
input_4         А
"К
input_5         А
"К
input_6         А
p

 
к "bЪ_
К
0         

К
1 
К
2         P
К
3         P╗
(__inference_model_layer_call_fn_40051329О(\]nohibcz{ДЕЪЫФХийвг─┼╛┐┼¤в∙
ёвэ
тЪ▐
#К 
inputs/0         А
#К 
inputs/1         А
#К 
inputs/2         А
#К 
inputs/3         А
#К 
inputs/4         А
#К 
inputs/5         А
p 

 
к "bЪ_
К
0         

К
1 
К
2         P
К
3         P╗
(__inference_model_layer_call_fn_40051395О(\]nohibcz{ДЕЪЫФХийвг─┼╛┐┼¤в∙
ёвэ
тЪ▐
#К 
inputs/0         А
#К 
inputs/1         А
#К 
inputs/2         А
#К 
inputs/3         А
#К 
inputs/4         А
#К 
inputs/5         А
p

 
к "bЪ_
К
0         

К
1 
К
2         P
К
3         P°
&__inference_signature_wrapper_40050805═(\]nohibcz{ДЕЪЫФХийвг─┼╛┐┼квж
в 
ЮкЪ
-
input_1"К
input_1         А
-
input_2"К
input_2         А
-
input_3"К
input_3         А
-
input_4"К
input_4         А
-
input_5"К
input_5         А
-
input_6"К
input_6         А"єкя
8
encoder1/fcn3'К$
encoder1/fcn3         P
8
encoder2/fcn3'К$
encoder2/fcn3         P
;
tf.__operators__.add_18 К
tf.__operators__.add_18 
<
tf.math.truediv_4'К$
tf.math.truediv_4         