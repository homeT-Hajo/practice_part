?E
??
8
Const
output"dtype"
valuetensor"
dtypetype

NoOp
?
PartitionedCall
args2Tin
output2Tout"
Tin
list(type)("
Tout
list(type)("	
ffunc"
configstring "
config_protostring "
executor_typestring 
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
@
ReadVariableOp
resource
value"dtype"
dtypetype?
?
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
executor_typestring ?
q
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape?"serve*1.15.02unknown8?8
l
save_counterVarHandleOp*
shape: *
dtype0	*
shared_namesave_counter*
_output_shapes
: 
e
 save_counter/Read/ReadVariableOpReadVariableOpsave_counter*
dtype0	*
_output_shapes
: 

NoOpNoOp
?
ConstConst"/device:CPU:0*?
valueyBw Bq
"
save_counter

signatures
IG
VARIABLE_VALUEsave_counter'save_counter/.ATTRIBUTES/VARIABLE_VALUE
 *
dtype0*
_output_shapes
: 
G
serving_default_xPlaceholder*
dtype0*
_output_shapes
:
?
PartitionedCallPartitionedCallserving_default_x**
f%R#
!__inference_signature_wrapper_688**
config_proto

GPU 

CPU2J 8*
Tout
2*,
_gradient_op_typePartitionedCallUnused*
Tin
2*
_output_shapes
:
O
saver_filenamePlaceholder*
dtype0*
_output_shapes
: *
shape: 
?
StatefulPartitionedCallStatefulPartitionedCallsaver_filename save_counter/Read/ReadVariableOpConst*%
f R
__inference__traced_save_715**
config_proto

GPU 

CPU2J 8*
Tout
2*,
_gradient_op_typePartitionedCallUnused*
_output_shapes
: *
Tin
2	
?
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filenamesave_counter*(
f#R!
__inference__traced_restore_730**
config_proto

GPU 

CPU2J 8*
Tout
2*,
_gradient_op_typePartitionedCallUnused*
_output_shapes
: *
Tin
2?-
?
?
__inference__traced_save_715
file_prefix+
'savev2_save_counter_read_readvariableop	
savev2_1_const

identity_1??MergeV2Checkpoints?SaveV2?SaveV2_1?
StringJoin/inputs_1Const"/device:CPU:0*<
value3B1 B+_temp_a3843b2959bc48daba4ae0fc6d927c6f/part*
dtype0*
_output_shapes
: 2
StringJoin/inputs_1?

StringJoin
StringJoinfile_prefixStringJoin/inputs_1:output:0"/device:CPU:0*
N*
_output_shapes
: 2

StringJoinZ

num_shardsConst*
value	B :*
dtype0*
_output_shapes
: 2

num_shards
ShardedFilename/shardConst"/device:CPU:0*
dtype0*
_output_shapes
: *
value	B : 2
ShardedFilename/shard?
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: 2
ShardedFilename?
SaveV2/tensor_namesConst"/device:CPU:0*<
value3B1B'save_counter/.ATTRIBUTES/VARIABLE_VALUE*
dtype0*
_output_shapes
:2
SaveV2/tensor_names?
SaveV2/shape_and_slicesConst"/device:CPU:0*
dtype0*
_output_shapes
:*
valueB
B 2
SaveV2/shape_and_slices?
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0'savev2_save_counter_read_readvariableop"/device:CPU:0*
_output_shapes
 *
dtypes
2	2
SaveV2?
ShardedFilename_1/shardConst"/device:CPU:0*
value	B :*
dtype0*
_output_shapes
: 2
ShardedFilename_1/shard?
ShardedFilename_1ShardedFilenameStringJoin:output:0 ShardedFilename_1/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: 2
ShardedFilename_1?
SaveV2_1/tensor_namesConst"/device:CPU:0*1
value(B&B_CHECKPOINTABLE_OBJECT_GRAPH*
dtype0*
_output_shapes
:2
SaveV2_1/tensor_names?
SaveV2_1/shape_and_slicesConst"/device:CPU:0*
valueB
B *
dtype0*
_output_shapes
:2
SaveV2_1/shape_and_slices?
SaveV2_1SaveV2ShardedFilename_1:filename:0SaveV2_1/tensor_names:output:0"SaveV2_1/shape_and_slices:output:0savev2_1_const^SaveV2"/device:CPU:0*
dtypes
2*
_output_shapes
 2

SaveV2_1?
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0ShardedFilename_1:filename:0^SaveV2	^SaveV2_1"/device:CPU:0*
T0*
N*
_output_shapes
:2(
&MergeV2Checkpoints/checkpoint_prefixes?
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix	^SaveV2_1"/device:CPU:0*
_output_shapes
 2
MergeV2Checkpointsr
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: 2

Identity?

Identity_1IdentityIdentity:output:0^MergeV2Checkpoints^SaveV2	^SaveV2_1*
_output_shapes
: *
T02

Identity_1"!

identity_1Identity_1:output:0*
_input_shapes
: : : 2(
MergeV2CheckpointsMergeV2Checkpoints2
SaveV2SaveV22
SaveV2_1SaveV2_1:+ '
%
_user_specified_namefile_prefix
?
.
__inference_forward_682
x
identityS
add/yConst*
valueB
 *  ??*
dtype0*
_output_shapes
: 2
add/yI
addAddV2xadd/y:output:0*
T0*
_output_shapes
:2
addL
IdentityIdentityadd:z:0*
T0*
_output_shapes
:2

Identity"
identityIdentity:output:0*
_input_shapes
::! 

_user_specified_namex
?
?
__inference__traced_restore_730
file_prefix!
assignvariableop_save_counter

identity_2??AssignVariableOp?	RestoreV2?RestoreV2_1?
RestoreV2/tensor_namesConst"/device:CPU:0*<
value3B1B'save_counter/.ATTRIBUTES/VARIABLE_VALUE*
dtype0*
_output_shapes
:2
RestoreV2/tensor_names?
RestoreV2/shape_and_slicesConst"/device:CPU:0*
valueB
B *
dtype0*
_output_shapes
:2
RestoreV2/shape_and_slices?
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*
dtypes
2	*
_output_shapes
:2
	RestoreV2X
IdentityIdentityRestoreV2:tensors:0*
T0	*
_output_shapes
:2

Identity?
AssignVariableOpAssignVariableOpassignvariableop_save_counterIdentity:output:0*
dtype0	*
_output_shapes
 2
AssignVariableOp?
RestoreV2_1/tensor_namesConst"/device:CPU:0*1
value(B&B_CHECKPOINTABLE_OBJECT_GRAPH*
dtype0*
_output_shapes
:2
RestoreV2_1/tensor_names?
RestoreV2_1/shape_and_slicesConst"/device:CPU:0*
dtype0*
_output_shapes
:*
valueB
B 2
RestoreV2_1/shape_and_slices?
RestoreV2_1	RestoreV2file_prefix!RestoreV2_1/tensor_names:output:0%RestoreV2_1/shape_and_slices:output:0
^RestoreV2"/device:CPU:0*
_output_shapes
:*
dtypes
22
RestoreV2_19
NoOpNoOp"/device:CPU:0*
_output_shapes
 2
NoOp{

Identity_1Identityfile_prefix^AssignVariableOp^NoOp"/device:CPU:0*
T0*
_output_shapes
: 2

Identity_1?

Identity_2IdentityIdentity_1:output:0^AssignVariableOp
^RestoreV2^RestoreV2_1*
T0*
_output_shapes
: 2

Identity_2"!

identity_2Identity_2:output:0*
_input_shapes
: :2$
AssignVariableOpAssignVariableOp2
RestoreV2_1RestoreV2_12
	RestoreV2	RestoreV2:+ '
%
_user_specified_namefile_prefix
?
8
!__inference_signature_wrapper_688
x
identity?
PartitionedCallPartitionedCallx* 
fR
__inference_forward_682**
config_proto

GPU 

CPU2J 8*
Tout
2*,
_gradient_op_typePartitionedCallUnused*
Tin
2*
_output_shapes
:2
PartitionedCall]
IdentityIdentityPartitionedCall:output:0*
_output_shapes
:*
T02

Identity"
identityIdentity:output:0*
_input_shapes
::! 

_user_specified_namex"?J
saver_filename:0StatefulPartitionedCall:0StatefulPartitionedCall_18"
saved_model_main_op

NoOp*x
serving_defaulte
 
x
serving_default_x:0%
output_0
PartitionedCall:0tensorflow/serving/predict*>
__saved_model_init_op%#
__saved_model_init_op

NoOp:?
N
save_counter

signatures
__call__"
_generic_user_object
:	 2save_counter
,
serving_default"
signature_map
?2?
__inference_forward_682?
???
FullArgSpec
args?
jx
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *?
	?
*B(
!__inference_signature_wrapper_688xm
!__inference_signature_wrapper_688H ?
? 
?

x?	
x"$?!

output_0?
output_0C
__inference_forward_682(?
?
?	
x
? "	?