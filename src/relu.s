.globl relu

.text
# ==============================================================================
# FUNCTION: Array ReLU Activation
#
# Applies ReLU (Rectified Linear Unit) operation in-place:
# For each element x in array: x = max(0, x)
#
# Arguments:
#   a0: Pointer to integer array to be modified
#   a1: Number of elements in array
#
# Returns:
#   None - Original array is modified directly
#
# Validation:
#   Requires non-empty array (length ≥ 1)
#   Terminates (code 36) if validation fails
#
# Example:
#   Input:  [-2, 0, 3, -1, 5]
#   Result: [ 0, 0, 3,  0, 5]
# ==============================================================================
relu:
    li t0, 1             
    blt a1, t0, error     
    li t1, 0             

    lw t0, 0(a0)
    li t2, 0
loop_start:
    # TODO: Add your own implementation
    bge t2, a1, end_loop  
    slli t5, t2, 2
    add t4, t5, a0
    lw t3, 0(t4) 
    blt t3, zero, set_zero
    addi t2, t2, 1
    j loop_start

set_zero:
    li t3, 0
    sw t3, 0(t4)
    addi t2, t2, 1
    j loop_start

end_loop:
    ret

error:
    li a0, 36          
    j exit       

