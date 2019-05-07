"""
Copyright (c) 2018, Intel Corporation

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

    * Redistributions of source code must retain the above copyright notice,
      this list of conditions and the following disclaimer.
    * Redistributions in binary form must reproduce the above copyright
      notice, this list of conditions and the following disclaimer in the
      documentation and/or other materials provided with the distribution.
    * Neither the name of Intel Corporation nor the names of its contributors
      may be used to endorse or promote products derived from this software
      without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
"""

############################################################################
# Purpose: Main AAD script for algorithm discovery
# Author: Ruchira Sasanka
# Date: Nov 2018
############################################################################

import random
import copy
import pickle
import multiprocessing as mp
import sys
import os
import signal
import re
import glob
import socket


############################################################################
# Various knobs (parameters) controlling the behavior of this framework
############################################################################
class Knob:

    # select which problem groups to solve. Usually make only one of these 
    # true. Knob group2solve can be set with command line arg too.
    #
    group2solve = 0               # use this if only 1 class is requested
    #
    solve_groupA = False          # use these if multiple classes required
    solve_groupB = False 
    solve_groupC = False 

    # evolution epochs and steps.
    #
    num_epochs = 100               # parallel epochs
    num_steps_per_epoch = 2000     # number of steps to evolve
    num_steps_per_class = 100      # steps for going in one class

    # Whether to end run when ALL problems are solved -- at least one solution
    # is available for each problem. If False, will run until end of epoch
    # count, num_epochs
    #
    end_when_solutions_found = True

    # Link and mutate attempts per function built
    #
    num_link_attempts = 2          # number of mutate tries per a func built
    num_opcode_mutates = 2         # number of opcode mutates per link attempt

    # We start by stetting the stmt limit to 'stmt_limit_start'. Then, after
    # every 'epochs_per_stmt_incr' epochs, we increment stmt limit by 1
    # NOTE: This is a "soft" limit. When we add exprs to a func under this
    # limit, if we decide to add an idiom (multiple exprs), this limit can 
    # be exceeded.
    #
    stmt_limit_start = 12           # starting stmt limit

    # increment stmt_limit by 1 after this many epochs
    #
    epochs_per_stmt_incr = 100 

    # Allow to randomly *lower* stmt count (stmt_limit_start), by this amount  
    #
    vary_stmt_cnt_by = 0            

    # extra stmts allowed for rebuilding from an existing solution
    #
    extra_stmts_for_rebuild = 2     

    # maximum number of solutions we record for a given problem like 'max'
    # 'min'. We keep multiple solutions around to increase diversity -- 
    # evolver can use them as seeds for new solutions. 
    #
    max_solns4prob = 100

    # Add some expressions to ExpIdiomStore on a coin toss. 
    # This should be between 0 and 100. Make this 100 to add unconditionally.
    # This option allows different parallel ranks to get a different mix of
    # expressions in each epoch
    #
    binop_add_odds = 80            # odds for adding binary operators
    pop_ind_add_odds = 20          # odds for adding pop at any index
    logic_op_add_odds = 10         # odds for adding 'or' 'and' 'not'
    #
    # Each stmt has a this much chance to undergo a mutation in Phase 3. 
    # Must be anumber between 0 and 100
    #
    stmtMutationOdds = 100

    # When a function is picked for rebuilding, clean it up by removing dead
    # code, this % of time
    #
    cleanUpOdds = 50

    # Pick an already built solution instead of creating one from scratch
    # this percent of time. Must be between 0 and 100. Value 0 will force
    # to build afresh always
    # Built solutions are picked from one's own class. 
    #
    builtSolnPickOdds = 40

    # Even when we have found a solution to a problem, this percent of time
    # we will find additional solutions, to increase solution diversity
    #
    find_more_solns_odds = 80

    # Out of solutions received by a rank, what percent is added as func
    # calls to be used in future solutions.
    # As we add more problems to solve, and as more solutions are found,
    # we have to limit the number of calls
    # NOTE: find_more_solns_odds also reduces sending solutions to ranks
    #
    funcCallAddOdds = 80           # same as exprAddOdds but for func calls
    #
    # The funcCallAddOdds controls only adding function calls received
    # at the start of an epoch, as a part of already found solutions.
    # The following knob prevents adding function calls that are 
    # discovered within the rank itself. This is for data gathering
    # purposes only to show the effectiveness of func composition
    #
    addDiscovedCallsWithinRank = True  

    # Allow this percent of time, function calls to be also considered for
    # addition, when picking an expression
    #
    fcall_pick_odds = 100

    # Odds of picking an idiom every time we insert an expression/idiom
    # This number can be between 0 and 100
    #
    idiomPickOddsBase = 10
    idiomPickOddsMult = 3

    # prefer inserting idioms early in function building
    #
    preferIdiomsEarly = 1

    # The percentage of time we should use enumerated for expression
    # like (for x,y in enumerated(arr)). Set this to 0 to skip this form
    # of for expression
    #
    enumerated_for_odds = 20

    # Detect infinite loops and recover
    #
    detectInfiniteLoops = True     # Detect infinite loops
  
    # Set this to 2 to create a process per *physical* CPU. If set to 1,
    # a process per each *logical* CPU will be created.
    #
    hyperthreads_per_cpu = 2

    # Use append expr with remove opcode as well. Remove is treated to be
    # a 2nd order operator because it contains a loop in it. Therefore, 
    # we should try to avoid it unless absolutely needed. 
    # 
    use_remove_expr = False

    # Analyze for dead PROD_CON arguments, when PROD args are not present
    # in an expression
    #
    analDeadPCon = False

    # If a test case failed after cleaning a func, assert
    # This assert is not a functional bug. This usually indicates an
    # inadequate checker
    #
    assert_on_testclean = False

    # EID numbering stars from this value
    #
    startingEID = 10

    # Maximum default problem size (for checking)
    #
    max_probsz = 200

    # When a solution is identified, print checker output between these
    # two limits
    #
    start_test_prn_iter = 10
    end_test_prn_iter = 30

    emitBlkMarkers = False         # whether to emit BLK_START/END for code

    # if a rank saw more than this many time-outs, exit the rank because
    # it could take a long time to complete and become a serial bottleneck
    #
    max_timeouts = 20

    # At the end of a simulation, print least complex/length solutions
    # for that run
    #
    print_least_complex_solns = True

    # In addition to least-complex solutions printed, compose 
    # solutions from least-complex solution found for *each*
    # function called in a solution. This is for reporting
    # only, and usually done with checkpoints
    # 
    compose_least_complex = False

    # Name of the report file produced at the end of a run 
    # (e.g., parents, individual least complex functions, etc) 
    #
    report_name = "report.out.txt"

    # Name of the full report file, processed from reading one or more
    # checkpoint files
    #
    full_report_name = "full_report.out.txt"

    # Checkpointing knobs
    # 1. Start entire run from a previously saved checkpoint. The checkpoint
    #    could have been taken on 1 or many nodes
    #
    start_from_chkpt = False
    #
    # 2. Dump checkpoints after 'epochs4chkpt' epochs. In a cluster env, each
    #    node will dump its own checkpoint
    #
    dump_chkpts = True             # dump checkpoint
    epochs4chkpt = num_epochs      # dump a delta chkpt every this many epochs
    #
    # 3. if we are running in a cluster environment, read checkpoints from
    #    other nodes after just dumping node's own checkpt. This is mainly
    #    done to exchange solutions found on each node. When reading, the
    #    latest checkpoint available from each node is read
    #
    read_cluser_chkpt = False
    #
    # 4. Directory where checkpoints are saved
    #
    chkpt_dir = "./chkpts"

    # Logging knobs
    #
    log_dir_root = "./log"         # log files for each rank produced in dir
    log_fname_root = "out"         # root name of output file of each rank


############################################################################
# Class defining debug constants -- These constants give a name to each
# debug region (e.g., a function, a block, stmt), we need to control for
# debug printing purposes. Set them to a non-zero value for debug printing
# We have them as integers rather than bools so we can have multiple levels
############################################################################
class Dbg:

    # Leave the following zero by default to reduce debug output
    #
    checker = 0
    manual_check = 0      # initial test of checker
    initFunc = 0
    buildFunc = 0
    linkMutate = 0
    execFunc = 0
    evalException = 0     # print on some 'exec' exception
    print_recursions = 0  # print aborted functions due to recursion
    dupFunctions = 0      # duplication function detection
    renameEID = 0
    packageFunc = 0
    finalCode = 0
    stepCnt = 0           # step counts
    evolver = 0
    evolveOnce = 0
    evolveNsteps = 0    
    detectDeadBlks = 0    # print when entire blks are dead code eliminated
    #
    testSoln = 0
    test1Soln = 0
    test1SolnSucc = 0
    test2Soln = 0

    # Leave these 1 by default to print results and checking
    #
    test3Soln = 1
    multi_proc = 1        # multi processor output
    chkpt = 1             # checkpoint save/restore

    # For printing a debug message, based on debug setting
    #
    def prn(dbg_val, msg):
        if (dbg_val > 0 ):
            print(msg)

    # For printing a debug message, based on debug setting, without line break
    #
    def prnwb(dbg_val, msg):
        if (dbg_val > 0 ):
            print(msg, end='')

    # unconditional warning message 
    #
    def warn(msg):
        print("**WARN**: ", msg)



############################################################################
# Global Stat class for common Global Stats
# This class has only class variables, so no objects are created out of this
#
############################################################################
class GlobStat:

    num_steps = 0                   # number steps
    epoch = 0                       # epoch number
    num_execs = 0                   # number executions (built programs)
    num_excpetions = 0              # number exceptions 
    num_recursions = 0              # number of recursion detected
    num_timeouts = 0                # number of timeout expiration

    def dump(pid):
        print("GlobStats (procid ", pid, "):",
              "epoch: ", GlobStat.epoch,
              "# steps: ", GlobStat.num_steps,
              "# execs: ", GlobStat.num_execs,
              "# exceptions: ", GlobStat.num_excpetions,
              "# recursion detections", GlobStat.num_recursions,
              "# time outs:", GlobStat.num_timeouts)

############################################################################
# Class defining a problem to be solved. It has an input list and output
# list. This is a base class and a specific problem (e.g., sort) must be
# derived from this class
# NOTE: A problem has more than one output,. A function can do that using
#       a tuple
# TODO: Make this an abstract class
############################################################################
class Problem:

    def __init__(self, name, def_sz, minsz, maxsz):
        self.name = name               # name of the problem
        #
        self.inputs = list()           # actual inputs
        self.input_types = list()      # type of each input
        #
        self.outputs = list()          # actual outputs
        self.output_types = list()     # type of outputs
        
        self.prod_cons_arg = None      # input arg#, if one is a PROD_CONS

        self.def_prob_size = def_sz    # default size to start testing
        self.min_prob_size = minsz     # minimum problem size
        self.max_prob_size = maxsz     # maximum problem size

        # Current problem size
        # This is used by the ProblemGenerators to generate an input
        # 
        self.prob_size = def_sz          # size of the current problem
        

    # Method to add an input type to the problem. Usually called when 
    # setting up the problem
    #
    def addInputType(self, inp_type): 
        self.input_types.append(inp_type)
        self.inputs.append(None)       # reserve a space in input arr

    # Method to add an output type to the problem. This is usually called
    # when a problem is defined.
    #
    def addOutputType(self, out_type): 
        self.output_types.append(out_type)
        self.outputs.append(None)      # reserve a space in output arr

    # set problem size
    #
    def setProblemSize(self, size):
        self.prob_size = size

    # initialize to minimum problem size
    #
    def initProbSize(self):
        self.prob_size = self.def_prob_size

    # Set actual input. Usually called later by a ProblemGenerator to 
    # set the input
    #
    def setInput(self, ind, inp): 
        self.inputs[ind] = inp

    # Method to set an actual output to the problem. Usually done later
    # when the generated algorithms can create output
    #
    def setOutput(self, ind, out): 
        self.outputs[ind] = out
    
    def setArgAsProdCons(self, arg_num):
        assert(arg_num == 0)            # only arg0 can be a PROD_CONS
        self.prod_cons_arg = arg_num 

    def getName(self):
        return(self.name)

    # print actual input output values for the purpose of printing
    #
    def getActInputOutputStr(self):
        out_str = "Inputs: "
        for inp in self.inputs:
            out_str += str(inp) + "\t"
        out_str += "Outputs: "
        for out in self.outputs:
            out_str += str(out) + "\t"
        return out_str


############################################################################
# The checker base class. A specific problem (e.g., sort) must be derived
# from this class and must implement a 'check' routine for checking the
# output for the given inputs in the problem
# TODO: Make this an abstract class
############################################################################
class Checker:
    def __init__(self, problem):
        self.prob = problem

    # All derived classes must over-ride this method to check problem.output
    # for input problem.inputs
    #
    def check(self):
        ThisMethodMustBeOverRidden


############################################################################
# The base class for generating a problem. A specific problem (e.g., sort) 
# must be derived from this class and must implement a generateProblem
# method to generate a problem of given size. 
# TODO: Make this an abstract class
############################################################################
class ProblemGenerator:
    def __init__(self, problem):

        self.prob = problem

    # derived classes must override this method to generate new inputs of
    # a given size
    #
    def generateNewInputs(self):
        ThisMethodMustBeOverRidden
 
############################################################################
# Base class for building expressions and statements. Since an expression
# has inputs and an output, this class models that -- i.e., types of args 
# and whether each arg is an input or/and output. There can be only one
# output from an expression, but since an arg can be read-modify-write
# we can have the output arg be both a producer and a consumer.
#
# TODO: Revisit only one output -- operations like 'split' produce two 
#       outputs
#
# The fields in an Expr are filled in two phases: Derived classes will 
# specify the type of each arg and the number of args. Objects created
# out of those derived types will represent an expression in a program --
# for this, we have expression IDs (eid's) that link one expression 
# to another. EIDs are filled in the 'evolution' phase.
#
# Each expression has the simplest possible form -- operation and args.
# Each arg can be coming from another expression. Such expressions are
# linked by expression IDs (eid's) as shown below:
#
# expr1 = [ .... input arr ... ]  # input arg1
# expr2 = 5                       # input arg2
# expr3 = expr2 * 2
# expr4 = append(expr1, expr3)
# expr5 = expr3 > 10
# if (expr5) {                   
#   expr6 = ....                 
#   expr7 = ....
# }
# expr6 = i > 10
# for x in tuple(expr1) {
#    ...

# }
#
#
#
# Note: Expressions are embedded in a in a Block Statement (BlkStmt), defined
#       later, to create program blocks
#
############################################################################
class Expr:

    # enumerations for data types supported in expressions
    # Must start at 0
    # WARN: Do not change the ordering of these. initFunc uses the order to
    #       detect the most complex type. Assertion fails if order changed.
    #
    BOOL = 0
    ELEM = 1             # element (scalar) is basically a number
    ARR =  2             # array (of elements)
    ARR_OF_ARR = 3       # array of arrays
    #
    TYPE_LAST = 4   # counts the # of prod types

    # Constant var prefix strings based on type
    #
    var_prefixes = ("bool_", "num_", "arr_", "arr_of_arr" )

    # enumerations for producer/consumer properties for dependence info
    #
    PROD = 1
    CONS = 2
    PROD_CONS = 3

    # Enumerations for special EIDs
    #
    EidOfRetVal = 1                     # EID of return value
    InvalEID = -1

    # Enumeration of the type of expression
    #
    EXPR_STMT = 1        # any expr that can be terminated with a semicolon
    CONTROL = 2          # CTRL_FLOW
    RET_VAL = 3          # return value of a function
    FUNC_ARG = 4         # argument of a function
    BLK_START = 5        # Block start marker
    BLK_END = 6          # Block end marker
    BLK_HEAD = 7         # for/if etc before a BLK_START (usu control flow)
    FUNC_CALL = 8        # Func call expression
    NULL = 9            # null (no effect) expression

    # Enumeration of emit mode
    #
    EMIT_PLAIN = 0       # emit as if we are not in a function
    EMIT_FUNC = 1        # emit to package into a function


    def __init__(self, args, prod_cons):
        self.arg_types = args           # tuple: type of each arg 
        self.prod_cons = prod_cons      # tuple: whether each arg is prod/con

        # print("NewExpr:", self.arg_types, self.prod_cons, len(prod_cons))

        # Sometimes, many related operations can be represented by the same
        # expression -- e.g., a binary operation. In such cases, they opcode
        # specifies which sub-operation is supported. This helps mutation as
        # well
        #
        self.opcode = 0                # current opcode
        self.num_opcodes = 0           # number of opcodes supported

        # TODO: Is this really necessary? We may need two outputs in some
        #       operations (e.g., in a 'split' operation). Even if we 
        #       supported multiple prods, we can use this an optimization
        #       for the common case. Let it be -1 for multi-prod exprs.
        #
        self.prod_pos = -1              # uint: pos of the prod in args tuple
        
        # number of producers and producer_consumers.  Usually, an 
        # expression has one producer or producer_consumer
        # However, we want to support expressions like enumerated for loops
        # in Python, which has multiple producers. Also, we may support 
        # operations like array splits, which has to producers.
        #
        self.num_prods = 0              # number of producers
        self.num_pcons = 0         # num of prod+prod_cons - updated below

        # When building, we will store expression ids of each arg here
        # TODO: Verify that producers or producer_consumers are listed first,
        # before any CONS
        #
        self.arg_eids = []              # arr: EID for each arg
        for pcon in prod_cons:
            self.arg_eids.append(Expr.InvalEID)
            if (pcon == Expr.PROD): 
                self.num_prods += 1
            elif (pcon == Expr.PROD_CONS):
                self.num_pcons += 1

        # indent level of this stmt. This is used for printing
        #
        self.indent_level = 0

        # The following describes the 'block path' of the *producer*. 
        # This is similar to section numbering of a chapter book -- 
        # e.g., Section 1.2.1, is a subsection of 1.2. Therefore, a we
        # know that a consumer in 1.2.1 can consume a producer in 1.2
        # but cannot consume a producer in 1.1. The path is stored as
        # a tuple. Level 0 does not have an entry -- i.e., empty tuple
        # describes level 0.
        # 
        self.blk_path = None

        # When this expr is in a BlkStmt (e.g., for an idiom), we can set
        # a hardness value, so it is not mutated easily
        #
        self.hardness = 0

        # Category for the expression. Each expression has a category and
        # some have special meanings (e.g., BLK_START, FUNC_CALL)
        #
        self.category = Expr.EXPR_STMT

        # Is this statement dead. We mark statements as dead in dead code
        # elimination.
        #
        self.is_dead = False

        # Usually, if an expr is not consumed, it is mark dead. However,
        # some control flow stmts (e.g., for/if) are not marked dead 
        # immediately. They are first marked not_cconsumed, and then marked
        # dead later after detecting there are no control flow effects
        # 
        self.not_consumed = False

        # Optional map, if set, used to emit an alternate eid for an 
        # existing eid in arg_eids[]. The emit function uses this
        #
        self.eid2s = None

        # set the emit mode. By default, it is outside of a func
        #
        self.emit_mode = Expr.EMIT_PLAIN


    # Method to set the producer position. The first PROC or PROD_CONS arg
    # is the producer position. This is a local method
    #
    def setProdPos(self):
        todo

    # Method to add an expression ID. When we are building a program, 
    # we will assign EID to each arg
    #
    def setArgEID(self, arg_num, arg_eid):
        self.arg_eids[arg_num] = arg_eid;

    # Method to emit the name of a give EID
    #
    def emitEID(self, arg_num):
        arg_eid = self.arg_eids[arg_num]          # existing eid for arg
        if (self.eid2s):                          # if expr has alt map
            new_eid = self.eid2s.get(arg_eid)     # if there is alt eid
            if (new_eid):
                arg_eid = new_eid                 # use that eid
        #
        var_prefix = Expr.var_prefixes[ self.arg_types[arg_num] ]
        return var_prefix + str(arg_eid)            # return the eid name

    # Method to emit the name of a give EID with Dbg Info
    #
    def emitEIDDbg(self, arg_num):
        return "eid_" + str(self.arg_eids[arg_num]) + \
            "(" + str(self.prod_cons[arg_num]) + ")";

    # Method to find out which args need an EID assignment (while building)
    # This returns -1 if all args have an EID assigned
    #
    def getNextMissingEID(self):
        todo
        # return arg_num of eid not assigned yet

    def setOpCode(self, opcode):
        self.opcode = opcode

    def setNumOpcodes(self, num):
        self.num_opcodes = num

    def markLive(self):
        self.is_dead = False
        self.not_consumed = False

    def markDead(self):
        self.not_consumed = True
        self.is_dead = True

    def markNotConsumed(self):
        self.not_consumed = True

    def isNotConsumed(self):
        return (self.not_consumed)

    def isDead(self):
        return (self.is_dead)

    def setBlkPath(self, counts, depth):
        self.blk_path = tuple(counts[0:depth])
        self.indent_level = depth

    def getBlkPath(self):
        return self.blk_path

    def isBlkStart(self):
        return (self.category == Expr.BLK_START)

    def isBlkEnd(self):
        return (self.category == Expr.BLK_END)

    def isFuncArg(self):
        return (self.category == Expr.FUNC_ARG)

    def isIfExpr(self):
        return False

    def isForExpr(self):
        return False

    def isBlkMarker(self):
        return ((self.category == Expr.BLK_END) or \
                (self.category == Expr.BLK_START) )
    
    def isBlkHead(self):
        return (self.category == Expr.BLK_HEAD)

    def isFuncCall(self):
        return (self.category == Expr.FUNC_CALL)

    def canMarkDead(self):
        return True


    # Set EID for all producers. Do not set for CONS or PROD_CONS args
    # Also, we set a producer ID, only if there are no current one
    #
    def setProdEIDs(self, start_eid):
        for i in range(len(self.arg_types)):            # for all args
            if (self.prod_cons[i] == Expr.PROD):        # if this is a producer
                if (self.arg_eids[i] == Expr.InvalEID): # set only if invalid
                    self.arg_eids[i] = start_eid
                    start_eid += 1
        return start_eid

    # Set the category
    #
    def setCategory(self, cat):
        self.category = cat

    # randomly change the opcode to a different one, if possible
    #
    def mutateOpcode(self):
        if (self.num_opcodes > 0):
            self.opcode = random.randint(0, self.num_opcodes - 1)


    # Method to emit the entire expression. This method is abstract.
    # Each derived class must have emit() routine defined
    #
    def emit(self):
        print("ERROR: This abstract method must have been overridden");
        abstract_method_called


    def isConsOrProdCons(self, pcon):
        return ((pcon == Expr.CONS) or (pcon == Expr.PROD_CONS))


#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# Append without any other opcodes. 
#
class AppendOnlyExpr(Expr):
    def __init__(self):
        argtypes = (Expr.ARR, Expr.ELEM)          # arg types
        pcons = (Expr.PROD_CONS, Expr.CONS)       # producer/consumer info
        super().__init__(argtypes, pcons)

    def emit(self):
        return super().emitEID(0) + ".append(" +  super().emitEID(1) + ")"




#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# Append expression with other opcode. E.g., 
#   arr1.append(element)
#   arr1.remove(element)
#
# Note: Remove is considered to be a 2nd order operation, because remove has
#       a loop inside. T
#
class AppendExpr(Expr):
    def __init__(self):
        argtypes = (Expr.ARR, Expr.ELEM)          # arg types
        pcons = (Expr.PROD_CONS, Expr.CONS)       # producer/consumer info
        super().__init__(argtypes, pcons)

        self.opcode_str = ('append', 'remove')
        #
        super().setNumOpcodes( len(self.opcode_str) )

    def emit(self):
        return super().emitEID(0) + "." +  self.opcode_str[ self.opcode ] + \
            "(" + super().emitEID(1) + ")"


#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# Arr Copy Expression (arr1 = arr2)
#
class ArrCopyExpr(Expr):
    def __init__(self):
        argtypes = (Expr.ARR, Expr.ARR)           # arg types
        pcons = (Expr.PROD, Expr.CONS)            # producer/consumer info
        super().__init__(argtypes, pcons)

    def emit(self):
        return super().emitEID(0) + " = " + super().emitEID(1) + ".copy()"


#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# Constant values
#
class ConstValExpr(Expr):
    def __init__(self):
        argtypes = (Expr.ELEM, )                  # arg types
        pcons = (Expr.PROD, )                     # producer/consumer info
        super().__init__(argtypes, pcons)
        #
        self.opcode_str = ('0', '1')             # constant values
        super().setNumOpcodes( len(self.opcode_str) )

    def emit(self):
        return super().emitEID(0) + " = " +  self.opcode_str[ self.opcode ]


#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# Pop expression. Pops last element from the array (modifies array)
#
class PopExpr(Expr):
    def __init__(self):
        argtypes = (Expr.ELEM, Expr.ARR)          # arg types
        pcons = (Expr.PROD, Expr.PROD_CONS)       # producer/consumer info
        super().__init__(argtypes, pcons)

        self.opcode_str = ('0', '')               # pop at end or start
        super().setNumOpcodes( len(self.opcode_str) )

    def emit(self):
        return super().emitEID(0) + " = " + super().emitEID(1) + ".pop(" + \
            self.opcode_str[ self.opcode ] + ")"

#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
class PopFirstExpr(Expr):
    def __init__(self):
        argtypes = (Expr.ELEM, Expr.ARR)          # arg types
        pcons = (Expr.PROD, Expr.PROD_CONS)       # producer/consumer info
        super().__init__(argtypes, pcons)

    def emit(self):
        return super().emitEID(0) + " = " + super().emitEID(1) + ".pop(0)"

#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
class PopIndExpr(Expr):
    def __init__(self):
        argtypes = (Expr.ELEM, Expr.ARR, Expr.ELEM)     # arg types
        pcons = (Expr.PROD, Expr.PROD_CONS, Expr.CONS)  # producer/consumer info
        super().__init__(argtypes, pcons)

    def emit(self):
        return super().emitEID(0) + " = " + super().emitEID(1) + ".pop(" + \
            super().emitEID(2) + ")"


#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
class PopFirstArrExpr(Expr):
    def __init__(self):
        argtypes = (Expr.ARR, Expr.ARR_OF_ARR)          # arg types
        pcons = (Expr.PROD, Expr.PROD_CONS)       # producer/consumer info
        super().__init__(argtypes, pcons)

    def emit(self):
        return super().emitEID(0) + " = " + super().emitEID(1) + ".pop(0)"


#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# First expression. Returns first (or last) element of a list 
#
class FirstExpr(Expr):
    def __init__(self):
        argtypes = (Expr.ELEM, Expr.ARR)          # arg types
        pcons = (Expr.PROD, Expr.CONS)            # producer/consumer info
        super().__init__(argtypes, pcons)

        self.opcode_str = ('0', '-1')             # constant values
        super().setNumOpcodes( len(self.opcode_str) )

    def emit(self):
        return super().emitEID(0) + " = " + super().emitEID(1) + "[" + \
            self.opcode_str[ self.opcode ] + "]"


#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#
class ArrOfArrCopyExpr(Expr):
    def __init__(self):
        argtypes = (Expr.ARR_OF_ARR, Expr.ARR_OF_ARR)          # arg types
        pcons = (Expr.PROD, Expr.CONS)            # producer/consumer info
        super().__init__(argtypes, pcons)

    def emit(self):
        return super().emitEID(0) + " = " + super().emitEID(1) + ".copy()"




#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# Expression representing an input arg. We need to specify the expression
# type for an input arg 
#
class InputArgExpr(Expr):
    def __init__(self, arg_type, arg_num):
        argtypes = (arg_type, ) 
        pcons = (Expr.PROD, )         # input arg is a producer           

        self.arg_num = arg_num        # argument number in a function

        super().__init__(argtypes, pcons)
        super().setCategory(Expr.FUNC_ARG)

        # The actual value of the arg, for the problem
        #
        self.arg_value = None


    # set the actual value of the argument
    #
    def setInputArgValue(self, value):
        self.arg_value = value

    # This method is called in two ways. When an input arg is emitted in 
    # main body, it is emitted as:
    #     expr_id = value
    #     expr_id = [..... array elements ....]
    # When it is emitted as a function argument, it is emitted as:
    #     expr_id = argX        # for a scalar
    #     expr_id = argX.copy() # for an array
    #     Note: copy() is used to keep the incoming arg constant
    #
    def emit(self):
        if (self.emit_mode == Expr.EMIT_FUNC):
            if (self.arg_types[0] == Expr.ARR):   # make arr arg constant
                return super().emitEID(0) + " = arg" + \
                    str(self.arg_num) + ".copy()"
            else:
                return super().emitEID(0) + " = arg" + str(self.arg_num) 

        else:
            return super().emitEID(0) + " = " + str(self.arg_value);
        

#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# A Compare Expression of the form: out = a > b
# This supports multiple opcodes: <, >, ==, <=, >=
#
class CmpExpr(Expr):
    def __init__(self):
        argtypes = (Expr.BOOL, Expr.ELEM, Expr.ELEM ) 
        pcons = (Expr.PROD, Expr.CONS, Expr.CONS )             
        super().__init__(argtypes, pcons)
        #
        self.opcode_str = ('<', '>', "==",'<=',  '>=', "!=")
        #
        super().setNumOpcodes( len(self.opcode_str) )


    def emit(self):
        out = super().emitEID(0) + " = (" + super().emitEID(1) + " " + \
            self.opcode_str[ self.opcode ] + " " +  super().emitEID(2) + ")"
        return out

#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# logical expressions of the form: out = expr1 and expr2
# Opcodes supported: or, and
#
class BinLogicalExpr(Expr):
    def __init__(self):
        argtypes = (Expr.BOOL, Expr.BOOL, Expr.BOOL ) 
        pcons = (Expr.PROD, Expr.CONS, Expr.CONS )             
        super().__init__(argtypes, pcons)
        #
        self.opcode_str = (' or ', ' and ')
        #
        super().setNumOpcodes( len(self.opcode_str) )


    def emit(self):
        out = super().emitEID(0) + " = " + super().emitEID(1) + \
            self.opcode_str[ self.opcode ] + " " + super().emitEID(2)
        return out



#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# logical expressions of the form: out = not expr
#
class UnaryLogicalExpr(Expr):
    def __init__(self):
        argtypes = (Expr.BOOL, Expr.BOOL ) 
        pcons = (Expr.PROD, Expr.CONS )             
        super().__init__(argtypes, pcons)

    def emit(self):
        out = super().emitEID(0) + " = not " + super().emitEID(1) 
        return out


#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# We derive Return statement from ExprStmt because only one expression is
# supported by a return statement (and there is no BLK)
#
class RetExpr(Expr):
    def __init__(self, ret_val_type):
        argtypes = (ret_val_type, )           # return value type
        pcons = (Expr.CONS, )                 # arg is a consumer
        super().__init__(argtypes, pcons)     # return expression with args
        super().setCategory(Expr.RET_VAL)

    # note: Instead of the 'return' keyword, we just assign to a variable
    #       called 'return_val', which will be set by exec()
    #
    def emit(self):
        if (self.emit_mode == Expr.EMIT_FUNC):
            return "return (" + super().emitEID(0) + ")"
        else:
            return "return_val = " + super().emitEID(0)


#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# Generic binary operator. Notice that the exact operation is determined
# by the opcode. This allows easy mutation (e.g., from addition to multip.)
#
class ReduceOpExpr(Expr):
    def __init__(self):
        argtypes = (Expr.ELEM, Expr.ELEM)
        pcons = (Expr.PROD_CONS, Expr.CONS)
        super().__init__(argtypes, pcons)

    def emit(self):
        out = super().emitEID(0) + " += " + super().emitEID(1)
        return out

    
#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# Generic binary operator. Notice that the exact operation is determined
# by the opcode. This allows easy mutation (e.g., from addition to multip.)
# 
class BinOpExpr(Expr):
    def __init__(self):
        argtypes = (Expr.ELEM, Expr.ELEM, Expr.ELEM)
        pcons = (Expr.PROD, Expr.CONS, Expr.CONS)
        super().__init__(argtypes, pcons)
        #
        self.opcode_str = ('+', '-', '*', '//')
        #self.opcode_str = ('+', '-')
        #
        super().setNumOpcodes( len(self.opcode_str) )

    def emit(self):
        out = super().emitEID(0) + " = " + super().emitEID(1) + " " + \
            self.opcode_str[ self.opcode ] + " " + super().emitEID(2)
        return out


#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# Dummy expression to mark a start of block (BLK_START)
#
class BlkStartExpr(Expr):
    def __init__(self):
        argtypes = tuple()
        pcons = tuple()
        super().__init__(argtypes, pcons)
        super().setCategory(Expr.BLK_START)

    def emit(self):
        if (Knob.emitBlkMarkers):
            return "#BLK_START"
        else:
            return ""
        

#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# Dummy expression to mark the end of block (BLK_END)
class BlkEndExpr(Expr):
    def __init__(self):
        argtypes = tuple()
        pcons = tuple()
        super().__init__(argtypes, pcons)
        super().setCategory(Expr.BLK_END)

    def emit(self):
        if (Knob.emitBlkMarkers):
            return "#BLK_END"
        else:
            return ""

#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
## General reduction operation. 
# NOTE: This shows the importance of PROD_CONS. With this we can write:
#  val9 = 0                
#  while () {
#     val9 += ...         # we can use val9 again as a PROD_CONS 
#  }
#
# TODO: Create an idiom like the above -- while loop with sum=0, and sum+=
#
class ReductionExpr(Expr):
    def __init__(self):
        argtypes = (Expr.ELEM, Expr.Elem)
        pcons = (Expr.PROD_CONS, Expr.CONS)
        super().__init__(argtypes, pcons)

        self.opcode_strs = ('+=', '-=', '*=', '/=')

    def emit(self):
        out = super().emitEID(0) + self.opcode_strs[ self.opcode ] + \
            super().emitEID(1)
        return out

#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
## Clears a variable (sets to 0)
#
class ClearExpr(Expr):
    def __init__(self):
        argtypes = (Expr.ELEM, )
        pcons = (Expr.PROD, )
        super().__init__(argtypes, pcons)

    def emit(self):
        out = super().emitEID(0) + " = 0"
        return out


#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# Creates new array
#
class NewArrExpr(Expr):
    def __init__(self):
        argtypes = (Expr.ARR, )
        pcons = (Expr.PROD, )
        super().__init__(argtypes, pcons)

    def emit(self):
        out = super().emitEID(0) + " = list()"
        return out
    

#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# Assignment operation between elements
#
class AssignExpr(Expr):
    def __init__(self):
        argtypes = (Expr.ELEM, Expr.ELEM )
        pcons = (Expr.PROD, Expr.CONS )
        super().__init__(argtypes, pcons)

    def emit(self):
        out = super().emitEID(0) + " = " + super().emitEID(1)
        return out


#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# This is an If expr (without block). This contains the bool condition check
# This is used as part of an IfStmt (and idiom)
#
class IfExpr(Expr):
    def __init__(self):
        argtypes = (Expr.BOOL, )                  # accepts a bool arg
        pcons = (Expr.CONS, )                     # arg is a consumer
        super().__init__(argtypes, pcons)         # create expr
        super().setCategory(Expr.BLK_HEAD)        # head of a BLK 
        self.name = "if"                          # short name
        
    def isIfExpr(self):
        return True

    def emit(self):
        return "if (" + super().emitEID(0) + "): " 

    # This expression must not be marked dead -- because there is a block
    # following it
    #
    def canMarkDead(self):
        return False


#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# For expression of the form: for x in arr
# This is used as part of a ForStmt (and idiom)
#
class ForExpr(Expr):
    def __init__(self):
        argtypes = (Expr.ELEM, Expr.ARR )         # for 'x' in 'arr'    
        pcons = (Expr.PROD, Expr.CONS )       
        super().__init__(argtypes, pcons)         # create expr
        super().setCategory(Expr.BLK_HEAD)        # head of a BLK 


    def emit(self):
        return "for " + super().emitEID(0) + " in tuple(" +  \
            super().emitEID(1) + "): " 

    def isForExpr(self):
        return True
    
    # The producer of a 'for stmtent' actually is in the next level.
    # That is when we say 'for x in ...', x is defined only in the block
    # following the ForExpr
    #
    #def setBlkPath(self, counts, depth):
    #    self.blk_path = tuple(counts[0:depth+1])
    #    self.indent_level = depth                 # indented at depth

    # This expression must not be marked dead -- because there is a block
    # following it
    #
    def canMarkDead(self):
        return False

    # Change the for expression to iterate over an arr_of_arr 
    #
    def changeToArrOfArr(self):
        argtypes = (Expr.ARR, Expr.ARR_OF_ARR )   # for 'arr' in 'arr_of_arr' 
        self.arg_types = argtypes



#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#
class ForExprEnum(Expr):
    def __init__(self):

        argtypes = (Expr.ELEM, Expr.ELEM, Expr.ARR )   # for i, elem in arr
        pcons = (Expr.PROD, Expr.PROD, Expr.CONS )
        super().__init__(argtypes, pcons)         # create expr
        super().setCategory(Expr.BLK_HEAD)        # head of a BLK 

    def emit(self):
        return "for " + super().emitEID(0) + ", " +  super().emitEID(1) + \
            " in enumerate(tuple(" + super().emitEID(2) + ")): "

    def isForExpr(self):
        return True

    # This expression must not be marked dead -- because there is a block
    # following it
    #
    def canMarkDead(self):
        return False

    # Change the for expression to iterate over an arr_of_arr
    #
    def changeToArrOfArr(self):
        argtypes = (Expr.ELEM, Expr.ARR, Expr.ARR_OF_ARR)
        self.arg_types = argtypes




#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# Function call expression. 
#
class FuncCallExpr(Expr):
    def __init__(self, func_name, ret_type, atypes, func_def_str, pcon_arg):

        pcons = list()                  # construct prod/consumer list
        argtypes = list()

        # Normal function call with a producer and all consumer args
        #
        if (pcon_arg == None):
            pcons.append(Expr.PROD)         # first arg is a producer
            argtypes.append(ret_type)       # type of producer

            for ty in atypes:         
                pcons.append(Expr.CONS)     # every arg in func call is a CONS
                argtypes.append(ty)         # append call argument types
        
        # This function call has no explicit PROD and has only a PROD_CONS
        #
        else:
            for argnum, ty in enumerate(atypes):         
                if (argnum == pcon_arg):
                    pcons.append(Expr.PROD_CONS)
                else:
                    pcons.append(Expr.CONS)
                argtypes.append(ty)    

            # Arg0 must be PROD_CONS, when there is no separate PROD
            # and arg0 must be the return type (output of the problem)
            #
            assert(pcon_arg == 0)
            assert(argtypes[0] == ret_type)


        #
        super().__init__(argtypes, pcons)         # create expr
        super().setCategory(Expr.FUNC_CALL)

        self.func_name = func_name                # func name
        self.num_args = len(atypes)               # num args to func
        assert(self.num_args > 0)
        self.func_def_str = func_def_str          # definition of function
        self.soln_class = None                    # class this func belongs to
        self.prod_cons_arg = pcon_arg             # if there is a prod_con_arg

    def emit(self):

        out1 = super().emitEID(0) + " = " + self.func_name + "("
        
        if (self.prod_cons_arg == None):                # if explicit prod
            out1 += super().emitEID(1)                  # emit arg 1
            for i in range(1, self.num_args):
                out1 += ", " + super().emitEID(i+1)     # emit other args
        else:                                           # if PROD_CONS
            out1 += super().emitEID(0)                  # emit arg 0
            for i in range(1, self.num_args):
                out1 += ", " + super().emitEID(i)       # emit other args

        out1 += ")"                                     # close func
        return out1









#############################################################################
# Base class for modeling one or more statements (block) -- e.g.,
# if, while statements contain statement blocks
#
# Note: IMPORTANT: The block (stmts) within the BlkStmt contains 'expression
#  statements' as in C. That is, an expression terminated by a semicolon. 
#  Therefore, each estmt is just an expression (Expr object). However, programs
#  contain nested statement blocks. To cater for that, we just have two 
#  special expression to mark start/end of a block. Therefore, estmts list
#  before contain only expressions. This decision is made so that we just 
#  have a flat list of expressions (compared to a recursive embedding of blks).
#  As a result, aAn estmt can be one of the following expressions:
#   1. A regular expression statement (e.g,, append)
#   2. A control expression (e.g., bool expr in if, while, else)
#   3: A BLK start/end expression (this is just a marker)
#   4: A func arg expression (an incoming arg to a func)
#   5: A return expression (return value of a function)
#   6. A Null expression (useful for mutation to nullify an expr)
#
# As a consequence of the above, when we insert a BlkStmt idiom into another
# BlkStmt (e.g., a func body), we just copy ctrl_expr plus all the estmts
# (which are just expressions), into the new BlkStmts' estmts.
#
# A program (function) is a BlkStmt
#
#############################################################################
class BlkStmt:
    def __init__(self):

        self.estmts = list()            # expression statements  in the BLK
        #

        # TODO: Make a function subclass and move these variables to them
        #
        # start numbering producer EIDs from this
        #
        self.cur_prod_eid = BlkStmt.StartEID

        # number of header expressions, if this represents a function
        #
        self.header_exprs = 0

        # The most complex incoming arg type. Must be set at function 
        # initialization
        #
        self.most_complex_arg_type = -1

    #------------------------------------------------------------------------
    # append an expr
    #
    def appendExpr(self, expr):        # append another expr statement to BLK
        self.estmts.append(expr)
 
    #------------------------------------------------------------------------
    # emit an entire block
    # Accepts and alternate mapping for eids, to 'merge' producers
    #
    def emit(self, eid2senior, func_mode=Expr.EMIT_PLAIN, head_str=""):

        fname2codestr = dict()          # map: function name => code str

        out_str = ""                    # output string from this func

        prev_estmt = None               # track prev stmt
 
        for estmt in self.estmts:       # for each statement

            indent_level = estmt.indent_level

            if (func_mode == Expr.EMIT_FUNC):     # for func mode 
                indent_level += 1                    # print additional tab

            for tab in range(indent_level):          # print tabs to indent
                out_str += "\t"
                
            # comment out dead stmts, except BLK_START/BLK_END
            #
            if (estmt.isDead()):         
                #out_str += "# DEAD: "          
                out_str += "#DEAD\n"
                #
                # assert that we cannot have dead blk markers in func mode
                #
                if (func_mode == Expr.EMIT_FUNC):
                    assert(not estmt.isBlkMarker())   
                #
                continue

            estmt.eid2s = eid2senior     # Specify the alt map for emit
            estmt.emit_mode = func_mode  # set emit mode
            #
            # If there is an empty block, just emit pass
            # Note: We can easily prevent empty block printing
            #
            if (estmt.isBlkEnd() and prev_estmt.isBlkStart()):
                out_str += "pass"

            #
            out_str += estmt.emit()      # emit expr
            out_str += "\n"
            #
            estmt.eid2s = None           # reset the alternate map

            if (not estmt.isDead()):
                prev_estmt = estmt

                if (estmt.isFuncCall()): # make a record of func calls seen
                    fname2codestr[estmt.func_name] = estmt.func_def_str


        # Now we have generated code for this block (func). If this func
        # called any other func, just append code for that func def
        #
        func_def_str = ""
        for (fname, code_str) in fname2codestr.items():
            
            # if fname is not already present in func_def_str, concat it
            #
            match = re.search("\s"+fname+"\(", func_def_str)
            if (not match):
                func_def_str += code_str + "\n"
            elif (Dbg.dupFunctions > 0):
                print("Omit dup func:", fname, ":", code_str)

        # return code for this func plus any other funcs used
        #
        return func_def_str + head_str + out_str
    

    #------------------------------------------------------------------------
    # populate a list of sets -- a set for each data type -- with producer
    # eids corresponding to that type
    #
    def getAListOfListsOfProdEIDsByType(self):
        #
        # first create a list of sets, one for each type
        #
        prods_of_type = list()
        
        for i in range(Expr.TYPE_LAST):                    # all producer types
            eids_of_a_type = list()                        # create a list
            prods_of_type.append( eids_of_a_type )         # append to outer

        for estmt in self.estmts:                           # for each stmt

            # print("estmt:" + estmt.emit())

            for i in range(len(estmt.arg_types)):           # for each arg

                arg_type = estmt.arg_types[i]               # arg type at pos i
                pcons = estmt.prod_cons[i]                  # arg pcons

                if (pcons == Expr.PROD):
                    prod_eid = estmt.arg_eids[i]            # prod at pos i
                    assert(prod_eid != Expr.InvalEID)
                    prods_of_type[arg_type].append(prod_eid)# add to right list

        return prods_of_type


    # Enumerations
    #
    StartEID = Knob.startingEID


# =============================== Idioms ====================================

# Derive each idiom from BlkStmt 

#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# If statement, with a control_expression and a block
#
class IfStmt(BlkStmt):

    def __init__(self):

        super().__init__()                         # init BlkStmt


    #------------------------------------------------------------------------
    # This method fills in the actual expressions in the idiom
    #
    def createIdiom(self, ei_store, mid_expr_type, mc_arg_type):

        explist = list()

        # First, create a boolExpr (i.e., out = a > b)
        # Note: since we set EIDS, we need to create a deep copy
        #
        bool_expr_ref = ei_store.getExprOfType(Expr.BOOL)
        assert(bool_expr_ref)
        bool_expr = copy.deepcopy(bool_expr_ref)
        #
        # Now, create an if expression, and link wit bool_expr
        #
        if_expr = IfExpr();
        #
        assert(bool_expr.prod_cons[0] == Expr.PROD)
        bool_prod_eid = 2000                      # random eid to link
        bool_expr.setArgEID(0, bool_prod_eid)
        if_expr.setArgEID(0, bool_prod_eid)

        # Now, just insert in order
        #                                         # e.g.,
        explist.append( bool_expr )           # out1 = a > b  
        explist.append( if_expr )             # if out1:
        explist.append( BlkStartExpr() )      # BLK_START

        # Get an expression to insert into the BLK
        #
        mid_expr = ei_store.getExprOfType(mid_expr_type)
        explist.append( copy.deepcopy(mid_expr) )        

        explist.append( BlkEndExpr() )        # BLK_END

        return explist
        

#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# For statement in Python of the form: for 'x' in 'arr':
#
class ForStmt(BlkStmt):

    def __init__(self):

        super().__init__()                         # init BlkStmt

    #------------------------------------------------------------------------
    # This method fills in the actual expressions in the idiom
    #
    def createIdiom(self, ei_store, mid_expr_type, mc_arg_type=Expr.ARR):
        #
        explist = list()

        for_expr = None

        # Use either enumerated for iterator or non-enumerated iterator
        #
        if (random.randint(1,100) <= Knob.enumerated_for_odds):
            for_expr = ForExprEnum()          # for enumerated expr
        else:
            for_expr = ForExpr()  

        # if we have to iterate over arr_of_arr (instead of arr), change
        # types of for_expr
        #
        if (mc_arg_type == Expr.ARR_OF_ARR):
            for_expr.changeToArrOfArr()

        explist.append( for_expr )             # For expression
        explist.append( BlkStartExpr() )       # BLK_START

        # Get an expression to insert into the BLK
        #
        mid_expr = ei_store.getExprOfType(mid_expr_type)
        explist.append( copy.deepcopy(mid_expr) )        

        explist.append( BlkEndExpr() )         # BLK_START

        return explist





#############################################################################
# This class contains both a list of idioms (of type BlkStmt) and a list of
# valid expressions (of type Exprs)
# TODO: We can create objects of this class and eliminate some idioms and
#       expressions to emulate different environments for evolution
#
class ExprIdiomStore:

    def __init__(self):

        self.exprs = list()        # a list of valid exprs (of type Exprs)
        self.idioms = list()       # a list of idioms (of type BlkStmts)
        
        # STEP 1: Add all expressions here
        #
        self.addAllExprs()

        # Create a list of lists -- each sub list containing a exprs of
        # a given type (w.r.t. producers)
        #
        self.exprs_of_prod_type = list()          # a list of lists
        #
        # Record the init expression counts we found for each producer
        # type. This is useful to distinguish initial expressions from 
        # call expressions added later
        #
        self.init_expr_cnts = list()

        # create each list of producer types
        #
        for ty in range(Expr.TYPE_LAST):          
            prod_list = list()
            self.findProdExprs(ty, prod_list)            
            self.exprs_of_prod_type.append(prod_list)
            #
            self.init_expr_cnts.append(len(prod_list))
            #
            print("Added ", len(prod_list), " for type: ", ty)

        # An expression of the following list can provide an output of
        # different type from a given input. For instance, such expr can
        # produce an ELEM from an ARR (e.g., pop(arr)). These types of
        # exprs are useful at the very beginning of a function, when only
        # one type of args are present.
        #
        self.arr_from_elem_exprs = self.findExprsOfTypes(Expr.ARR, Expr.ELEM)
        self.elem_from_arr_exprs = self.findExprsOfTypes(Expr.ELEM, Expr.ARR)
        

        # STEP 2: Add all idioms here
        # NOTE: Don't move this up. Idioms can be added only after
        #       initializing 'self' variables above
        #
        self.addAllIdioms()


    #------------------------------------------------------------------------
    # Only add expressions based on a probability
    #
    def addRandom(self, expr, percent):
        should_add = (random.randint(1, 100) <= percent)
        if (should_add):
            self.exprs.append(expr)
            

    #------------------------------------------------------------------------
    # Add all exprs except InputArgExpr and RetExpr. They are added in 
    # a special way
    #
    def addAllExprs(self):
        #
        exprs = self.exprs

        # Add append expression based on knob. 
        #
        if (Knob.use_remove_expr):
            exprs.append( AppendExpr() )        # append, remove
        else:
            exprs.append( AppendOnlyExpr() )    # append only

        exprs.append( ArrCopyExpr() )
        exprs.append( NewArrExpr() )
        exprs.append( CmpExpr() )
        exprs.append( AssignExpr() )
        exprs.append( PopExpr() )
        exprs.append( FirstExpr() )
        exprs.append( ArrOfArrCopyExpr() )
        exprs.append( ConstValExpr() )

        # Add the following based on a coin toss. This allows different
        # parallel ranks to have different expression mixes. Only do this 
        # to expressions which are not absolutely necessary. 
        #
        self.addRandom( BinOpExpr(), Knob.binop_add_odds )
        self.addRandom( PopIndExpr(), Knob.pop_ind_add_odds )

        # Experiment with this
        #
        self.addRandom( ReduceOpExpr(), 10 )
        #self.addRandom( UnaryLogicalExpr(), Knob.logic_op_add_odds )
        #self.addRandom( BinLogicalExpr(), Knob.logic_op_add_odds )



    #------------------------------------------------------------------------
    # Function calls are added to the corresponding list of expressions
    # TODO: Optional: We can add 'fcall_expr' to a list to keep track
    #
    def addFuncCallExpr(self, fcall_expr):
        ret_type = fcall_expr.arg_types[0]
        self.exprs_of_prod_type[ret_type].append(fcall_expr)

    #------------------------------------------------------------------------
    # Add all idioms
    #
    def addAllIdioms(self):
        #
        idioms = self.idioms
        #
        idioms.append( IfStmt() )
        idioms.append( ForStmt() )
                   
    #------------------------------------------------------------------------
    # Returns an expression of given producer type
    #
    def getExprOfType(self, prod_type):

        prod_list = self.exprs_of_prod_type[prod_type]
        num_prods_of_type = len(prod_list)
        if (num_prods_of_type > 0):
            
            # By default, pick only from expressions added initially -- i.e.,
            # when this ExprIdiom store was initialized. Based on odds, 
            # we allow picking from any expression, including function calls
            #
            avail_exprs = self.init_expr_cnts[prod_type]
            #
            if (random.randint(1,100) < Knob.fcall_pick_odds):
                avail_exprs = num_prods_of_type

            expr_ind = random.randint(0, avail_exprs-1)     # pick expr
            return prod_list[expr_ind]

        else:
            print("No expressions of type found for type: ", prod_type)
            assert(0)        # We must have an expression of each type
            return None

    #------------------------------------------------------------------------
    # returns a list of expressions that produce a given type
    #
    def findProdExprs(self, prod_type, expr_list):
        
        # Go through each expression
        #
        for expr in self.exprs:

            for arg in range(len(expr.arg_types)):

                # is this the correct producer type
                #
                corr_pcon = (expr.prod_cons[arg] == Expr.PROD) or \
                            (expr.prod_cons[arg] == Expr.PROD_CONS)

                if (expr.arg_types[arg] == prod_type) and corr_pcon:
                    
                    expr_list.append(expr)
                    break

    #------------------------------------------------------------------------
    # returns a list of expressions that match the following:
    #    all producers are of type 'prod_type' AND
    #    all consumers are of type 'cons_type'
    # Used to find out 
    #
    def findExprsOfTypes(self, prod_type, cons_type):
        
        expr_list = list()

        for expr in self.exprs:

            mismatch = False                 # arg of a different type
            prod_found = cons_found = False  # whether prod/cons found

            # Make sure that all producers match the prod_type and all 
            # consumers match the cons_type
            #
            for arg in range(len(expr.arg_types)):

                if (expr.prod_cons[arg] == Expr.PROD):      # if prod
                    if (expr.arg_types[arg] == prod_type):  
                        prod_found = True
                    else:
                        mismatch = True
                else:
                    if (expr.arg_types[arg] == cons_type):  # if cons
                        cons_found = True
                    else:
                        mismatch = True

            # After going through all args, if 
            if (prod_found and cons_found and (not mismatch)):
                expr_list.append(expr)

        return expr_list


    #------------------------------------------------------------------------
    # This method returns a list of expressions that can be added to the
    # function under construction
    #
    def getExprsToInsert(self, pick_idiom, prod_type, func):

        if (pick_idiom):                    # if an idiom is requested

            # First, pick an idiom at random
            #
            num_idioms = len( self.idioms )
            assert(num_idioms > 0)         # must have some idioms
            idiom_pos = random.randint(0, num_idioms-1)
            idiom_picked = self.idioms[ idiom_pos ]

            # Create the actual contents (exprs) of the idiom based on the 
            # prod_type and most complex input arg type of func
            #
            mc_arg_type = func.most_complex_arg_type
            ret_exprs = idiom_picked.createIdiom(self, prod_type, mc_arg_type)

            if (Dbg.linkMutate > 0):
                print("Piked idiom # exprs:", len(ret_exprs))
            #
            #for e in estmts_of_idiom:
            #    exp = copy.deepcopy(e)  # make deep copy of Expr
            #    ret_exprs.append( exp ) # add each expr of idiom to 'ret_exprs
            #
            return ( ret_exprs )

        
        else:
            # pick an expr to add (based on type) and copy to 'ret_exprs' list
            #
            expr_picked = self.getExprOfType(prod_type)
            assert(expr_picked)
            
            #if (expr_picked.isFuncCall()):
                #print("Inserting call to ", expr_picked.func_name)
                #if (expr_picked.func_name == "DotProd"):
                #    Dbg.finalCode = 1

            #
            exp = copy.deepcopy(expr_picked)      # make deep copy of Expr
            assert(len(exp.arg_types) == len(expr_picked.arg_types))
                   
            ret_exprs = list()                    # list of exprs
            ret_exprs.append( exp )

            return ( ret_exprs )
                   

                  
#############################################################################
# This class is used to keep useful stats about a solution we find
#
class SolStats:
    def __init__(self):

        self.prob_name = None           # name of the problem
        self.step = -1                  # at which step was the soln found
        self.class_step = -1            # at which class step "
        self.pahse = -1                 # at which phase was the soln found 
        self.link_iter = None           # at which link iter  " 
        self.mutate_iter = None         # at which mutate iter "
        self.parent_name = None         # parent, if evolved from another sol
        self.used_clean = False         # used the cleaned version of sol
        self.sol_epoch = None           # soln found in this epoch
        self.stmt_cnt = None            # stmt count
        self.clean_stmt_cnt = None      # stmt count without dead code
        self.sol_procid = None          # sol found on this rank
        self.sol_time = None            # timestamp of solution


    def dump(self):
        print("STAT: ",
              " problem::", self.prob_name,
              " epoch:", self.sol_epoch,
              " step:", self.step,
              " class step:", self.class_step,
              " phase:", self.phase,
              " link_iter:", self.link_iter,
              " mutate_iter:", self.mutate_iter,
              " parent_name:", self.parent_name,
              " used_clean:", self.used_clean,
              " stmt_cnt:", self.stmt_cnt,
              " clean_stmt_cnt:", self.clean_stmt_cnt,
              " sol procid:", self.sol_procid,
              " sol_time:", self.sol_time)


#############################################################################
# Class for a packaged problem and solution.
# A solution is a packaged function that is grammatically correct and
# can produce output from input.
# After initializing, we can send an object of this class to an Evolver
# to complete the solution.
# Once a solution is built, we will record it in a list for future use.
#
# Inputs:
#   problem: a problem describing inputs, output and their types
#   problem_generator: generates inputs and output
#   checker: checks whether a generated output in problem is correct of not

class ProbSolnPack:
    def __init__(self, problem_generator, checker):

        self.problem = problem_generator.prob
        self.prob_gen = problem_generator
        self.checker = checker

        # Solution (func) found for the problem. Evolution algo will set
        # this after a solution is found
        # Note: We keep the correct eid2senior map for a function, rather
        #       than permanently adjusting eids to correct ones. This is done
        #       in order to keep SSA form in estmts, allowing the same 
        #       mutation algorithms to be applied in SSA form 
        #
        self.sol_func = None            # function (BlkStmt)
        self.sol_clean_func = None      # cleaned function of solution     
        self.sol_func_call_expr = None  # function call expression for func
        self.sol_eid2senior = None      # eid2senior mapping for soln
        self.sol_stats = None           # stats about solution found
        self.is_soln_new = False        # is solution newly found
        self.sol_metrics = None         # metrics for this soln

    def getName(self):                  # helper function to get name of prob
        return (self.problem.getName())


#############################################################################
# Inputs:
#   all_prob_soln: all packaged problem to be solved (from all classes)
#   idioms: all idioms and grammar (as a list)
#   incorr_solns: the pool of all incorrect solutions -- 
# Output:
#   prob_soln.func will be completed
#   Compilable, but incorrect, solutions added to incorr_solns
#
#   prob_soln_list contains problems with similar inputs and outputs (types)
#
class Evolver:

    def __init__(self, all_prob_soln_list, ei_store, epoch):

        # Record all problem/solution units from all classes to be solved
        #
        self.all_prob_soln_list = all_prob_soln_list

        # Problem/solution list of a SINGLE class. Temporarily, set it to
        # the first class. We iterate through all classes 
        #
        self.prob_soln_list = self.all_prob_soln_list[0]

        # First problem/solution unit of the class.
        # Until we find a solution, we use the very first ProbSolnPack
        # in prob_soln_list[] to refer to the solution, because all the
        # problems in prob_soln_list belong to the same class -- i.e., has
        # the same type of inputs/outputs
        #
        self.prob_soln = self.prob_soln_list[0]  

        # Expression and Idiom store
        #
        self.ei_store = ei_store

        # This is the function that we will construct and test throughout
        # this class
        #
        self.func = None

        # list of ProbSolnPack's in prob_soln_list, with valid soln_funcs
        # -- i.e., problems we have already found solutions for
        #
        self.all_solns_found = list()

        # This contains all classes this rank has solved. 
        #
        self.classes_solved = set()          # all classes solved

        # The following is when prob_soln_list already contains solutions
        # This can happen if we restart from a saved solution list
        # Also, find out which classes are already solved -- in a prior
        # epoch. 
        #
        for classid, psol_class_list in enumerate(all_prob_soln_list):   

            solns_found_class = list()               # add a list for class
            self.all_solns_found.append(solns_found_class)

            entire_class_solved = True               # whether entire class

            for psol in psol_class_list:             # for each psol in class
                if (psol.sol_func != None):
                    pname = psol.problem.getName()
                    Dbg.prn(Dbg.evolver, "Recording known soln for: " + pname)
                    solns_found_class.append(psol)    #add to proper class 

                    # even when a solution is present, add the function call
                    # only randomly. This controls complexity growth due to
                    # increasing solutions
                    #
                    do_add = (random.randint(1, 100) <= Knob.funcCallAddOdds)
                    if (do_add):
                        Dbg.prn(Dbg.evolver, "\r-recorded fcall for " + pname)
                        ei_store.addFuncCallExpr(psol.sol_func_call_expr)

                else:
                    entire_class_solved = False

            # if all the problems of this class are solved, add the classid
            #
            if entire_class_solved:
                self.classes_solved.add(classid)


        # Init to very first class. This will be set again in evolve
        #
        self.solns_found = self.all_solns_found[0]
        #
        self.cur_sol_class = 0                        # class id of prob/soln

        # Keep stats about the function under construction in this class
        # If we find a solution, we record these stats in the solution
        #
        self.func_stats = SolStats()

        # Whether this evolver seeing too many time outs
        #
        self.too_many_timeouts = False

        GlobStat.epoch = epoch

        # LATER: For later optimization
        #
        self.shellFunc = None           # For Optional optimization 


    #------------------------------------------------------------------------
    # Evolve for N steps, through all classes. 
    # We go through a few steps in each class and switch to the other class,
    # and repeat this process. We do this to allow cross-pollination -- i.e.,
    # solutions found in one class may be used in the solutions for other
    # classes. 
    #
    def evolveNSteps(self, num_steps, stmt_limit):

        # Number of steps to go through a class before switching to the 
        # other class
        #
        class_steps = Knob.num_steps_per_class

        for step in range(0, num_steps, class_steps):

            self.func_stats.step = step       # update stats

            # Go through all problem classes
            #
            #
            for class_id in range(len(self.all_prob_soln_list)):

                prob_class = self.all_prob_soln_list[class_id]

                # if the class is empty, nothing to do
                #
                if (len(prob_class) == 0):
                    Dbg.prn(Dbg.evolveNsteps, "Class ", class_id, " is empty")
                    continue

                # If we have solved the entire class, skip
                #
                if (class_id in self.classes_solved):
                    continue

                # print STEP info
                #
                if (Dbg.stepCnt > 0):
                    print("CLASS: ", class_id, " STEP: ", str(step) + ": ....")

                # Evolve a single class for some steps
                #
                self.prob_soln_list = prob_class
                self.cur_soln_class = class_id

                # Set solns_found class variable so that, 
                # if any solutions found, we will add to the proper class
                #
                self.solns_found = self.all_solns_found[class_id]
                #
                class_succ = self.evolveClassNSteps(class_steps, stmt_limit)
                
                # if the whole class was solved successfully, record it
                #
                if (class_succ):

                    self.classes_solved.add(class_id)  # record this class
                    print("added class: ", class_id, " set:", 
                          self.classes_solved) 

                    #
                    if (len(self.classes_solved) == \
                        len(self.all_prob_soln_list)):

                        print ("!!! SUCCESS: ALL CLASSES SOLVED")
                        return True

                # if this rank is seeing too many timeouts, just return
                #
                if (self.too_many_timeouts):
                    print("Warn: Too many time outs! Stopping class step") 
                    return False

        return False


    #------------------------------------------------------------------------
    # Carry out multiple evolution steps for a single class of problems
    # PRE: self.prob_soln_list must be set to the required class
    # 
    def evolveClassNSteps(self, num_steps, stmt_limit):

        #self.initFunc()
        #self.shellFunc = copy.deepcopy(self.func)
        
        # initialize all problem sizes and generate new inputs
        # Set self.prob_soln to first unsolved problem
        #
        saw_first_unsolved = False
        #
        for psol in self.prob_soln_list:
            if (psol.sol_func == None):           # if no soln yet

                psol.problem.initProbSize()       # init problem
                psol.prob_gen.generateNewInputs()

                if (not saw_first_unsolved):      # select first unsolved
                    self.prob_soln = psol
                    saw_first_unsolved = True

        # the class must have at least 1 unsolved problem
        #
        assert(saw_first_unsolved)
        assert(self.prob_soln.sol_func == None)

        # go through all steps
        #
        for step in range(num_steps):

            self.func_stats.class_step = step       # update stats
            GlobStat.num_steps += 1
            
            Dbg.prn(Dbg.stepCnt, "Class Step " + str(step) + ": ....")

            # Evolve one step
            #
            success = self.evolveOnce(stmt_limit)

            if (success):                          # if success, count solns
                
                num_solns = 0

                for psol in self.prob_soln_list:

                    if (psol.sol_func != None):
                        num_solns += 1

                # if we have found solutions to all the problems,
                # end evolution fro class and return true
                #
                if (num_solns == len(self.prob_soln_list)):
                    print("SUCCESS for entire CLASS !!!")
                    return True

                # Get out of this class. We need to find the next unsolved
                # psol, which we do at the start of this function. So, this
                # func must be called again
                #
                break

            # if this rank is seeing too many timeouts, just return
            #
            if (self.too_many_timeouts):
                Dbg.prn(Dbg.evolveOnce, "Warn: Too many time outs! Returning")
                return False


        # We exhausted the number of iterations and didn't find a solution
        #
        return False



    #------------------------------------------------------------------------
    # Try one evolution step. 
    #
    def evolveOnce(self, stmt_limit):

        # If a random adjustment to stmt_limit requested, do now
        #
        if (Knob.vary_stmt_cnt_by > 0):
            stmt_limit -= random.randint(0, Knob.vary_stmt_cnt_by)
            assert(stmt_limit > 0)


        # PHASE 1: ..... Build function ......................

        self.func = None                # function to be built
        #
        eid2path = dict()               # map for recording path of each prod

        self.func_stats.parent_name = None        # rest stat
        self.func_stats.used_clean = False        # rest stat

        # decide whether to build a new func or use an existing solution
        #
        if (len(self.solns_found) > 0):
            pick_soln = (random.randint(1, 100) <= Knob.builtSolnPickOdds)
            if (pick_soln):
                #
                # Randomly pick the index of function to pick
                #
                ind = random.randint(0, len(self.solns_found)-1)
                psol = self.solns_found[ind]      # soln at index
                #
                pname = psol.problem.getName()       # name of solution
                Dbg.prn(Dbg.evolveOnce, "Picking avail soln: " + pname) 

                # Decide whether to use cleaned up version of this func.
                # This gives more opportunity for a pick solution to be 
                # rebuilt
                #
                use_clean = (random.randint(1,100) <= Knob.cleanUpOdds)

                if (use_clean):
                    Dbg.prn(Dbg.evolveOnce, "\t-using cleaned up func") 
                    self.func = copy.deepcopy(psol.sol_clean_func)
                    self.func_stats.used_clean = True       # set stat
                else:
                    # Not using cleaned up copy
                    #
                    self.func = copy.deepcopy(psol.sol_func) # copy sol func
                #
                self.func_stats.parent_name = pname  # update stats
                assert(self.func != None)

                # We would allow some extra stmts for rebuilding a solution
                #
                stmt_limit += Knob.extra_stmts_for_rebuild

                if len(self.func.estmts) < stmt_limit:
                    Dbg.prn(Dbg.evolveOnce, "\t-will be re-building func")

        # if we are not using a solution, create and initialize func
        #
        if (self.func == None):
            self.initFunc() 
            #self.func = copy.deepcopy(self.shellFunc)

        # if function needs to be built (or rebuilt)
        #
        if (len(self.func.estmts) < stmt_limit):
            self.buildFunc(stmt_limit, eid2path)

        else:
            # a function already exists. We need to recreate eid2path,
            # to continue with linkAndMutate, because eid2path is filled
            # buildFunc
            #
            self.fillEID2Path(eid2path)


        # PHASE 2: ...... Link producers & Consumers .................
        # 
        test_success = False
        #
        self.func_stats.phase = 2       # update stats

        psol_found = None               # 

        # First, link the the producers and consumers for the first time.
        # If that does not succeed, try different linking
        #
        for step in range(Knob.num_link_attempts):

            eid2senior = dict()         # create a new dictionary

            should_mutate = (step > 0)  # Mutate after 1st step

            built = self.linkAndMutate(should_mutate, eid2senior, eid2path)
            #
            if (built):

                self.func_stats.link_iter = step      # update stats
                self.func_stats.mutate_iter = None    

                # Now, test thoroughly, whether the built func succeeds a large
                # number of test cases. 
                #
                psol_found = self.testSolution(eid2senior)
                test_success = (psol_found != None)
                if (test_success):
                    break

                # PHASE 3: Mutate once and test thoroughly
                #
                self.func_stats.phase = 3              # update stats
                #
                for mu in range(Knob.num_opcode_mutates):

                    self.func_stats.mutate_iter = mu   # update stats
                    #
                    self.mutateParameters()
                    #
                    psol_found = self.testSolution(eid2senior)
                    test_success = (psol_found != None)
                    if (test_success):
                        break
                
                if (test_success):
                    break


        # If all the testing succeeded (at least for one problem) 
        # we package it as a solution to be used in the future
        #
        if (test_success):
            self.packageFunc(eid2senior, psol_found)

        # See whether this rank is seeing too many timeouts
        #
        if (GlobStat.num_timeouts > Knob.max_timeouts):
            self.too_many_timeouts = True


        return test_success

    #------------------------------------------------------------------------
    # Remove dead blocks repeatedly
    #
    def removeDeadExprs(self, func, eid2senior):

        num_attempts = 3                # do up to 3 nested blocks

        for cnt in range(num_attempts):  
            removed_blk = self.removeDeadExprsOnce(func, eid2senior)
            if (not removed_blk):
                break


    #------------------------------------------------------------------------
    #
    def removeDeadExprsOnce(self, func, eid2senior):
        
        # STEP1: First detect dead *inner* blocks. If all the stmts inside
        # an inner block is dead, and any of the producer of head (e.g., for)
        # is not consumed, then we mark the head as dead as well
        # Note: We usually do not mark head exprs (like for, if) as dead
        #       because they are control flow expressions. However, if the
        #       entire block is dead and head is dead itself, we can get rid
        #       of it as well
        #
        head_estmt = None                         # head detected
        blk_start_estmt = None                    # blk start detected

        num_live_estmts_in_blk = 0
        saw_dead_blk = False

        for estmt in func.estmts:                 # for each estmt

            if estmt.isBlkHead():                 # record block head
                head_estmt = estmt
                blk_start_estmt = None

            elif estmt.isBlkStart():              # reset stmt cnt
                blk_start_estmt = estmt
                num_live_estmts_in_blk = 0

            elif estmt.isBlkEnd():                # at block end
                if num_live_estmts_in_blk == 0:   # if all estmts dead in blk

                    assert(head_estmt != None)
                    assert(blk_start_estmt != None)

                    # if head is not consumed OR if this is an If expression
                    # with an empty block, we can eliminate it
                    # NOTE: If expressions don't have a producer so they are
                    #       not marked not_consumed. 
                    #
                    if (head_estmt.isNotConsumed() or head_estmt.isIfExpr()):
                    #if (head_estmt.isNotConsumed()):
                        #                             
                        head_estmt.markDead()     
                        blk_start_estmt.markDead()
                        estmt.markDead()
                        saw_dead_blk = True
                        num_live_estmts_in_blk = 1   # disable until BLK_START
            else:
                if (not estmt.isDead()):
                    num_live_estmts_in_blk += 1


        if (Dbg.detectDeadBlks and saw_dead_blk):
            print("Removed Dead Block. Code: -----\n", func.emit(eid2senior))


        # STEP: Now, append non-dead estmts to a new list
        #
        new_estmts = list()             # create a new list
        
        for estmt in func.estmts:

            # if not dead append to new list
            #
            if not estmt.isDead():    
                new_estmts.append(estmt)
            else:
                # A function argument cannot be dead
                #
                assert(not estmt.isFuncArg())
        
        func.estmts = new_estmts        # use the new list
        assert(len(func.estmts) > 0)    # everything cannot be dead

        return saw_dead_blk

    #------------------------------------------------------------------------
    #
    def removeDeadExprsOLD(self, func, eid2senior):

        # STEP: Now, append non-dead estmts to a new list
        #
        new_estmts = list()             # create a new list
        
        for estmt in func.estmts:
            if (not estmt.isDead()):    # if not dead, append to new list
                new_estmts.append(estmt)

        func.estmts = new_estmts        # use the new list
        assert(len(func.estmts) > 0)    # everything cannot be dead


    #------------------------------------------------------------------------
    # Go through each stmt and records its producer path in eid2path
    # 
    def fillEID2Path(self, eid2path):

        func = self.func                # func of the soln

        # Go thru each estmt in function 
        #
        for estmt in func.estmts:

            # Go through each arg in estmt (expr)
            #
            for arg in range(len(estmt.prod_cons)):

                # get the type of the arg, its current eid, and prod_cons
                #
                arg_eid = estmt.arg_eids[arg]
                arg_prod_cons = estmt.prod_cons[arg]
                
                if (arg_prod_cons == Expr.PROD):
                    eid2path[arg_eid] = estmt.getBlkPath()


    #------------------------------------------------------------------------
    # Generate new input values and set them in Problem
    #
    def setNewInputValues4Prob(self, psol, psize):

        # set the size of the problem and generate new inputs
        #
        prob = psol.problem             # shortcut to Problem object in psol

        prob.setProblemSize(psize)
        #
        psol.prob_gen.generateNewInputs()


    #------------------------------------------------------------------------
    # Generate new input values and set them in Problem and InputArgExpr 
    # estmts in function under construction (self.func)
    #
    def setNewInputsAndFuncArgs(self, psol, psize):
        
        # First set input values
        #
        self.setNewInputValues4Prob(psol, psize)
                
        # Set the new input args in the function under construction
        #
        prob = psol.problem
        #
        for arg in range(len(prob.inputs)):
            
            in_arg_expr = self.func.estmts[arg]             # get input arg
            in_arg_expr.setInputArgValue(prob.inputs[arg])  # set new val


    #------------------------------------------------------------------------
    # Test the function under construction (self.func) is a solution for
    # any of the problems we have in prob_soln_list
    #
    def testSolution(self, eid2senior):
        
        psol_found = None

        # Take the problem of the very first ProbSoln pack in 
        #
        prob0 = self.prob_soln.problem

        # Execute the function once and set output
        #
        exec_ret = self.execFunc(prob0.getName(), eid2senior)
        if (exec_ret == None):
            return None

        assert(len(prob0.outputs) == 1) # currently supports only 1 output

        prob0.setOutput(0, exec_ret)    # set output of problem

        if (Dbg.testSoln > 0):
            print("exec ret: ", exec_ret, " outs:", prob0.outputs)

        check_passed = False

        # Now, go through all checkers in prob_soln_list and see whether
        # any checker succeeds
        #
        for psol in self.prob_soln_list:

            # if there is already a solution for this problem, skip
            # TODO: Maintain a separate list in class, which needs solution
            #       finding. Then we don't need this check
            #
            if (psol.sol_func != None):
                continue

            # For this 'psol' (ProbSolnPcck), set the same inputs and
            # outputs of 'prob0', since we don't want to create them again
            #
            prob = psol.problem
            prob.inputs = prob0.inputs
            prob.outputs = prob0.outputs

            # Do an assertion check to make sure problems are the same
            #
            if(prob0.output_types[0] != prob.output_types[0]):
                print("Prob0:", prob0.getName(), "!= Prob:", prob.getName())
                assert(0)

            # We should be testing with default problem size
            #
            assert(prob.prob_size == prob.def_prob_size)

            # Now, call the checker of psol, and see whether it passes
            #
            if (psol.checker.check()):
                check_passed = True
                if (Dbg.test1SolnSucc > 0):
                    print("***Check1 passed for problem " + prob.getName())
                    print("---Code:---\n", self.func.emit(eid2senior))
                break
            else:
                Dbg.prn(Dbg.test1Soln, prob.getName() + ": check failed")

        # if none of the checkers passed, just return False
        #
        if (not check_passed):
            return None


        # we are here because at least one checker passed. So, check each
        # solution. This time, we are going to generate new inputs/outputs
        # for each problem and check
        # Note: it is important that we go through all the tests -- at least
        #       all the ones after the one that succeeded Test1, because 
        #       some other checker later may pass every test.
        #
        for psol in self.prob_soln_list:

            # if there is already a solution for this problem, skip
            # TODO: Maintain a separate list in class, which needs solution
            #       finding. Then we don't need this check
            #
            if (psol.sol_func != None):
                continue

            prob = psol.problem         # problem in psol

            # Do test2. Try different input with same problem size
            #
            test2_passed = True
                #
            for st in range(1,2):   # TODO: magic number
            
                # Set inputs, execute, and check
                #
                self.setNewInputsAndFuncArgs(psol, prob.def_prob_size)
                exec_ret = self.execFunc(prob.getName(), eid2senior)
                if (exec_ret == None):
                    test2_passed = False
                    break

                prob.setOutput(0, exec_ret)
                #
                if (not psol.checker.check()):     # if check failed
                    test2_passed = False
                    break

            # if test2 passed, this looks like a probable solution. So,
            # now check for multiple input sizes
            #
            if (test2_passed):

                if (Dbg.test2Soln > 0):
                    print()
                    print("***Check2 passed for problem " + prob.getName())
                    print("---Code:---\n", self.func.emit(eid2senior))

                test3_passed = True

                start = prob.min_prob_size
                end = prob.max_prob_size       
                #
                for psize in range(start, end):   # try different prob sizes
          
                    # Set inputs, execute, and check
                    #      
                    self.setNewInputsAndFuncArgs(psol, psize)
                    exec_ret = self.execFunc(prob.getName(), eid2senior)
                    if (exec_ret == None):
                        test3_passed = False
                        break

                    prob.setOutput(0, exec_ret)
                    #
                    if (not psol.checker.check()):
                        Dbg.prn(Dbg.test2Soln, " ... but Check3 failed!!\n")
                        test3_passed = False
                        break
                    elif (psize >= Knob.start_test_prn_iter):                
                        # print inputs/outputs
                        #
                        prn_end = Knob.end_test_prn_iter

                        if ((Dbg.test3Soln > 0) and (psize < prn_end)):
                            print("Pass: ", prob.getActInputOutputStr())
                                        
                # if test3 passed as well, we accept this solution
                #
                if (test3_passed):
                    if (Dbg.test2Soln > 0):
                        print()
                        print("!!!Check3 ALSO passed for: " + prob.getName())
                        print("SUCCESS!!!")    
                    return psol
                else:
                    # Check3 failed. So, restore default prob size
                    # After we go through different problem sizes, we have to
                    # restore the original size
                    # Note: If Check3 passed, we will package this function
                    #       and we will reset this when we clean up
                    #
                    self.setNewInputsAndFuncArgs(psol, prob.def_prob_size)


        # We are here because we did not find any checker accepting the 
        # solution. So, return false
        #
        return None

    #------------------------------------------------------------------------
    #
    def testCleanedFunc(self, newfunc, psol, eid2senior):
        
        # Take the problem of the very first ProbSoln pack in
        #
        prob = psol.problem                       # problem of psol
        save_func = self.func                     # save the cur func
        self.func = newfunc                       # use new func to exec
        
        # execute with max problem sz and assert -- packaged func cannot fail
        #
        self.setNewInputsAndFuncArgs(psol, prob.max_prob_size)
        exec_ret = self.execFunc(prob.getName(), eid2senior)
        
        if (exec_ret == None):
            print("Problem:", psol.problem.getName())
            print("ERROR: Orig: ---\n", save_func.emit(eid2senior))
            print("ERROR: Cleaned: --\n", self.func.emit(eid2senior))

        if (Knob.assert_on_testclean):
            assert(exec_ret != None)

        # set output and check -- check must pass
        #
        prob.setOutput(0, exec_ret)
        chkout = psol.checker.check()

        if (not chkout):
            print("Problem:", psol.problem.getName())
            print("ERROR: Orig: ---\n", save_func.emit(eid2senior))
            print("ERROR: Cleaned: --\n", self.func.emit(eid2senior))

        if (Knob.assert_on_testclean):
            assert(chkout)

        self.func = save_func                     # restore func
        
        # set default problem size and regenerate input
        #
        self.setNewInputsAndFuncArgs(psol, prob.def_prob_size)


    #------------------------------------------------------------------------
    # Initialize the function with arguments and the return value
    #
    def initFunc(self):

        self.func = BlkStmt()                     # a func is a block statement
        func = self.func                          # 
        problem = self.prob_soln.problem
        #
        Dbg.prn(Dbg.initFunc, "creating func args")

        # First add the arguments as producers to func.stmts
        #
        arg = 0
        func.header_exprs = 0
        missing_arg_types = {Expr.ELEM, Expr.ARR} # init main arg types

        for arg_type in problem.input_types:           # for each arg in prob
            in_arg_expr = InputArgExpr(arg_type, arg)  # create input arg expr
            #
            # Set producer EID for each arg
            #
            func.cur_prod_eid = in_arg_expr.setProdEIDs(func.cur_prod_eid)
            #
            # record the actual arg value
            #
            in_arg_expr.setInputArgValue(problem.inputs[arg])
            arg += 1
            #
            func.appendExpr(in_arg_expr)     # add input arg expr to func
            func.header_exprs += 1

            # remove arg types present, so 'missing_arg_types' will contain
            # only actually missing arg types
            #
            if arg_type in missing_arg_types:
                missing_arg_types.remove( arg_type )

            # Detect the most complex input arg type used in this function.
            # This depends on the ordering of types, and the assert checks
            # this ordering
            #
            assert((Expr.ARR > Expr.ELEM) and (Expr.ARR_OF_ARR > Expr.ARR))
            #
            if (func.most_complex_arg_type < arg_type):
                func.most_complex_arg_type = arg_type


        # Add a conversion expression to convert an argument to
        # a different type (Optional). E.g., from list 
        # ExprIdiomStore.arr_from_elem_exprs
        # If this is  not done, our mutate algo may fail in some cases
        
        for miss_arg_ty in missing_arg_types:

            expr_list = None

            if (miss_arg_ty == Expr.ELEM):
                expr_list = self.ei_store.elem_from_arr_exprs
            elif (miss_arg_ty == Expr.ARR):
                expr_list = self.ei_store.arr_from_elem_exprs

            num_exprs = len(expr_list)
            assert(num_exprs > 0)
            expr_ind = random.randint(0, num_exprs - 1)
            expr_ref = expr_list[ expr_ind ]
            expr = copy.deepcopy( expr_ref )

            func.cur_prod_eid = expr.setProdEIDs(func.cur_prod_eid)

            func.appendExpr(expr)
            #
            func.header_exprs += 1

        # Third, we add the return expression to func.stmts. The return
        # stmt has just a consumer -- so not necessary to set any prod eid.
        #
        ret_val_expr = RetExpr(problem.output_types[0])
        self.func.appendExpr(ret_val_expr)

        if Dbg.initFunc > 0:
            print("Initial func:-------- \n", self.func.emit(dict()))
          

    #------------------------------------------------------------------------
    # PHASE 1:
    # Phase 1: build a function with exprs and idioms. After building, there
    #          are missing links. Therefore, we must use linkAndMutate()
    #          to complete func
    # Inputs:
    #    -- stmt_limit: max # of stmts to be used in the function
    #
    def buildFunc(self, stmt_limit, eid2path):

        func = self.func                # func of the soln

        # Record all producers of func, grouped in sets. Each producer is 
        # inserted into a list of its corresponding type. 'eids_of_type' 
        # is a list of those lists. This 'list of lists of producers' is
        # used to quickly lookup producers of a given type, for linking
        # and mutations. This 'eids_of_type' is updated as we insert 
        # more statements and traverse up the func, below.
        #
        eids_of_type = func.getAListOfListsOfProdEIDsByType()

        # We will not add statements beyond a given threshold
        #
        reached_max_stmts = False

        # index (from the end of list) pointing to the current estmt
        # This is used to go thru func.estmts in reverse order
        #
        stmt_index = 0

        blk_depth = 0                   # block depth (from root)

        # count of blocks at each block depth. For creating an implicit
        # dominator tree path
        #
        blk_depth_counts = []

        dbg_cnt = stmt_limit * 2         # for debug


        # go thru all estmts in func in *reverse* order -- i.e., starting
        # from return statement and going up. We use a 'while True' 
        # construct because new stmts get added to func.estmts, within this
        # loop. So, a standard for loop is not viable
        #
        while True:

            dbg_cnt -= 1                # detect any logic errors
            assert(dbg_cnt >= 0)

            # Get the next estmt, going up from the bottom
            #
            num_estmts_in_func = len(func.estmts)

            # Move to the next stmt (from the bottom)
            # This points to the stmt, whose sources are examined to insert
            # new exprs/idioms
            #
            stmt_index += 1

            # If we "processed" all the statements, we succeeded in building
            # of a function. Processing involves building eid2path as well
            #
            if (stmt_index > num_estmts_in_func):
                break

            # get the next statement
            #
            estmt = func.estmts[num_estmts_in_func - stmt_index]

            # Whether we reached the limit of number of stmts allowed
            # We do not break out immediately if we reached the max stmts,
            # because we allow linking to happen for any remaining stmts
            # 
            reached_max_stmts = (num_estmts_in_func >= stmt_limit)

            # Update the indent level and producer path of each stmt
            #
            if (estmt.isBlkEnd()):     

                blk_depth += 1                              # entered new blk

                if (blk_depth >= len(blk_depth_counts)):    # append entry
                    blk_depth_counts.append(0)

                blk_depth_counts[blk_depth-1] += 1          # incr counter
            #
            estmt.setBlkPath(blk_depth_counts, blk_depth)   # set path/level
            #
            if (estmt.isBlkStart()):                        # exited a blk
                blk_depth -= 1

            if (Dbg.buildFunc > 0):
                print("Stmt[", stmt_index, " P:", estmt.blk_path, "]: ", \
                          estmt.emit())

            # Go through each arg in estmt (expr)
            #
            for arg in range(len(estmt.prod_cons)):

                # get the type of the arg, its current eid, and prod_cons
                #
                arg_type = estmt.arg_types[arg]
                arg_eid = estmt.arg_eids[arg]
                arg_prod_cons = estmt.prod_cons[arg]

                # if the arg is a producer, remove it from the eids_of_type
                # because this stmt or any stmt above (in func) cannot use
                # this producer to link to a consumer
                # ASSUMPTION: Producers are listed first in an expr
                #
                if (arg_prod_cons == Expr.PROD):

                    if (Dbg.buildFunc > 0):
                        print("\t-removing prod: ", arg_eid)

                    eids_of_type[arg_type].remove(arg_eid)
                    eid2path[arg_eid] = estmt.getBlkPath()    # record path

                # if the argument is invalid, which means it has not
                # been linked yet. This can be true only for CONS and 
                # PROD_CONS 
                #
                if (arg_eid == Expr.InvalEID):

                    assert(estmt.isConsOrProdCons( arg_prod_cons ) ) 

                    num_estmts_in_func = len(func.estmts)

                    reached_max_stmts = (num_estmts_in_func >= stmt_limit)

                    reached_headers = \
                        stmt_index > (num_estmts_in_func - func.header_exprs)

                    if ((not reached_headers) and (not reached_max_stmts)):
                        
                        # We can still add more statements. 

                        # Decide whether to insert a single expression (expr
                        # stmt) or an idiom
                        #
                        rand_num = random.randint(1, 100)

                        # Prefer idioms at the start of the building process
                        # This allows more expressions to get into blocks
                        #
                        odds = Knob.idiomPickOddsBase 
                        odd_mult = Knob.idiomPickOddsMult

                        if (Knob.preferIdiomsEarly > 0):
                            odds += (stmt_limit - num_estmts_in_func)*odd_mult

                        pick_idiom = (rand_num <= odds) 

                        new_exprs = \
                            self.ei_store.getExprsToInsert(pick_idiom, \
                                                           arg_type, func)

                        # Give new EIDs to producers and add them to 
                        # eids_of_type. Also, rename any existing
                        # EIDs to new ones.
                        #
                        func.cur_prod_eid = \
                            self.renameAndAddArgEIDs(new_exprs,  \
                                                     func.cur_prod_eid, \
                                                     eids_of_type)

                        # insert 'new_exprs' at a "suitable" place in func.
                        #
                        # we can insert only above stmt_index, but below
                        # header_exprs
                        #
                        # First, calculate how many 'insert_points' are 
                        # available between header_exprs and top of 
                        # stmt_idex
                        #
                        assert(stmt_index < num_estmts_in_func)
                        ins_points = num_estmts_in_func - stmt_index
                        assert(ins_points >= func.header_exprs)
                        ins_points -= func.header_exprs

                        # Now, pickup a random position ('ins_pos') between 
                        # the bottom of header_exprs and top of stmt_index
                        #
                        rand_ins_pos = random.randint(0, ins_points)
                        #
                        ins_pos = func.header_exprs + rand_ins_pos 
                        assert(ins_pos < num_estmts_in_func)

                        # if 'ins_pos' is just above a BLK_START, we should
                        # move 1 pos down
                        #
                        if (func.estmts[ins_pos].isBlkStart()):
                            ins_pos += 1
                        assert(ins_pos < num_estmts_in_func)

                        # Finally, insert to estmts
                        #
                        for nexp in reversed(new_exprs):
                            func.estmts.insert(ins_pos, nexp)

                        # Optional TODO: If we insert an expr, we can link
                        # The prod/cons too. 

                        if (Dbg.buildFunc > 0):
                            print("Num exprs inserted:",  len(new_exprs), \
                                      " stmt count:", len(func.estmts))
                        
                            
        if (Dbg.buildFunc > 0):
            print("Built Func:--- \n", self.func.emit(dict()))



    #------------------------------------------------------------------------
    # PHASE 2:
    # 2nd phase: complete any missing links and optionally mutate
    # The same algo is used for both completing missing links and mutating
    # based on the args given
    #
    # Side effects:
    #   + self.func get EIDs assigned
    #   + eid2senior: fills dictionary mapping an eid (with level > 0) 
    #     to a 'senior' producer eid (upstream in the same branch). 
    #
    # Note: We use eid2senior dictionary to map multiple *producers* to the
    #       same 'variable name' of a program. Without this, each producer
    #       is unique, resulting in a single static assignment (SSA). Thus,
    #       eid2senior achieves a form of 'phi elimination' by mapping 
    #       different producers among multiple paths to the same name
    #       Since, this mapping is done using this separate 'eid2senior' map,
    #       this gives us another mutation opportunity, to simply 
    #       change the map, without doing any changes to eids in exprs
    #
    def linkAndMutate(self, should_mutate, eid2senior, eid2path):

        func = self.func                # func of the soln

        # Record all producers of func, grouped in sets. Each producer is
        # inserted into a list of its corresponding type. 'eids_of_type'
        # is a list of those lists. This 'list of lists of producers' is
        # used to quickly lookup producers of a given type, for linking
        # and mutations. This 'eids_of_type' is updated as we traverse 
        # up the func.
        #
        eids_of_type = func.getAListOfListsOfProdEIDsByType()

        # Map  to keep track of which producers are consumed by later
        # estmts. For a given producer eid, this map gives the most
        # upstream level that  it is consumed in. This map is used to: 
        #  (1) detect dead producers, and
        #  (2) to do 'phi elimination' 
        #
        prod_eid_consumed_path = dict()

        stmt_num = 0                    # stmt number (from bottom)

        func_success = True             # whether we built a successful func

        # Go thru each estmt in function in *revere* order -- i.e., from
        # return statement to func args
        #
        for estmt in reversed(func.estmts):

            estmt.markLive()            # by default, this estmt is live
            prods_not_consumed = 0      # number of producers not consumed 
            pcons_not_consumed = 0      # number of prod_con not consumed 

            stmt_num += 1               # move to next stmt
            stmt_success = True         # assume success
            blk_path = estmt.blk_path   # blk path of the producer

            if (Dbg.linkMutate > 0):
                print("Stmt[", stmt_num, " P:", estmt.blk_path, "]: ", \
                          estmt.emit())

            # Go through each arg in estmt (expr)
            #
            for arg in range(len(estmt.prod_cons)):

                # get the type of the arg, its current eid, and prod_cons
                #
                arg_type = estmt.arg_types[arg]
                arg_eid = estmt.arg_eids[arg]
                arg_prod_cons = estmt.prod_cons[arg]

                # Decide whether to mutate this arg on a coin toss, if 
                # mutation is requested
                # TODO: Make these magic numbers class members
                #
                rand_int = random.randint(1, 100)
                coin_toss_success = rand_int > 50
                mutate_arg = should_mutate #and coin_toss_success

                # if a consumer (or prod_cons) eid needs to entered and marked 
                # as consumed, it is set here, until we are sure that
                # estmt is alive
                #
                cons_eid_found = None

                # if the arg is a producer, remove it from the eids_of_type
                # because this stmt or any stmt above (in func) cannot use
                # this producer to link to a consumer
                # ASSUMPTION: Producers are listed first in an expr
                #
                if (arg_prod_cons == Expr.PROD):

                    eids_of_type[arg_type].remove(arg_eid)

                    # if this producer arg is not consumed below, increment
                    # producer not consumed. If all producers are not consumed
                    # then mark this stmt as dead. Don't worry about linking
                    # consumers of this (dead) estmt.
                    #
                    cons_blk_path = prod_eid_consumed_path.get(arg_eid)

                    if ( cons_blk_path == None ):     # if no consumer
                        prods_not_consumed += 1
                        if (prods_not_consumed == estmt.num_prods):
                            if ( estmt.canMarkDead() ):
                                estmt.markDead()

                                # if a func arg is dead, soln is  useless
                                #
                                if (estmt.isFuncArg()):
                                    return False
                                else:
                                    break             # done with estmt
                            else:  
                                estmt.markNotConsumed()
                                continue              # done with prod
                        else:
                            continue                  # done with prod

                    # if this producer does not dominate the consumer
                    # we need to link this producer with another 
                    # producer at a dominating level, for 'phi elimination' 
                    #
                    if (not self.canConsume(blk_path, cons_blk_path)):

                        if (Dbg.linkMutate > 0):
                            print("Need a senior prod for eid: ", arg_eid)

                        eid_list = eids_of_type[arg_type]
                        #
                        senior_prod = \
                            self.getEIDFromUpstream(eid_list, eid2path, \
                                                     cons_blk_path, False)

                        # enter a mapping, if we found a senior_prod
                        #
                        if (senior_prod != Expr.InvalEID):

                            # before we enter mapping for arg_eid, see whether
                            # some other junior is pointing to arg_eid as 
                            # a senior. In that case, we need to map that 
                            # junior to senior_prod as well
                            #
                            for junior, senior in eid2senior.items():
                                if (senior == arg_eid):
                                    eid2senior[junior] = senior_prod

                            # enter mapping  for arg_eid
                            #
                            eid2senior[arg_eid] = senior_prod

                            if (Dbg.linkMutate > 0):
                                print("\t-found senior prod: ", senior_prod)

                            # Mark the *senior_prod* as consumed, because
                            # now senior_prod cannot go dead
                            #
                            self.markProdAsConsumed(senior_prod, blk_path, 
                                                    prod_eid_consumed_path)

                        else:
                            # This could happen for BOOLs since we don't
                            # explicitly add a BOOL producer at the outer
                            # most level
                            #
                            Dbg.prn(Dbg.linkMutate, "Senior prod not found")
                            #stmt_success = False
                            #break                    

                # This arg is a consumer or prod_cons
                # Link consumers with producers. We MUST link if the cons
                # is not previously linked. We decide to mutate an existing
                # consumer on a coin toss (if mutation is requested)
                #
                elif ((arg_eid == Expr.InvalEID) or mutate_arg):

                    # If we are at the bottom of the function, we should
                    # use later producers -- otherwise, those producers become
                    # dead because there are no consumers further down
                    #
                    no_random_prod = (stmt_num <= 2) and (arg <= 1)

                    # Find a suitable producer based on the type of the 
                    # consumer. Here, the producer can be from any level
                    # giving a lot of flexibility in linking. If we end up
                    # linking a producer at a higher level, we will insert
                    # 'phi elimination' mapping for that producer (code above)
                    #
                    eid_list = eids_of_type[arg_type]
                    #
                    eid_picked = self.getEIDFromAnyBlk(eid_list, 
                                                       no_random_prod)

                    # If we can't find a producer of the same type to link 
                    # with, we cannot compile.
                    # Note: we can't mark this stmt dead because we will not
                    #       get here, if the prod of this stmt was dead -- i.e.
                    #       already there is a consumer below this stmt
                    #
                    if (eid_picked == Expr.InvalEID):
                        stmt_success = False
                        break

                    # Now the consumer has an eid coming from s producer.
                    # So, link them
                    #
                    estmt.setArgEID(arg, eid_picked)      # link prod/cons

                    # add the producer eid_picked to the set of consumed,
                    # if and only if the producer is not upstream of the
                    # existing consumer
                    #
                    cons_eid_found = eid_picked
                
                else:
                    # This is a consumer/prod_consumer with a valid arg_eid
                    # This can be true for idioms. Add to consumed dict.
                    # if and only if the level of consumer is lower than
                    # what is recorded for the producer 'eid_picked'
                    #
                     cons_eid_found = arg_eid


                # STEP: Detect dead PROD_CONS args. 
                # PROD_CONS args are set an eid above with a producer. However
                # that producer may not be consumed downstream. In that case
                # this PROD_CONS is dead. 
                # NOTE: We detect dead PROD args above. However, we have to
                #       wait until PROD_COSNS are assigned an eid to detect
                #       whether that eid is dead
                #
                if (Knob.analDeadPCon and (estmt.num_prods == 0) and \
                        (arg_prod_cons == Expr.PROD_CONS)):

                    # We need to see cons_eid_found is consumed 
                    # down stream
                    #
                    arg_eid = cons_eid_found

                    # if this prod_con arg is not consumed below, increment
                    # producer not consumed. If all producers are not consumed
                    # then mark this stmt as dead. Don't worry about linking
                    # consumers of this (dead) estmt.
                    #
                    cons_blk_path = prod_eid_consumed_path.get(arg_eid)

                    if ( cons_blk_path == None ):     # if no consumer
                        pcons_not_consumed += 1
                        if (pcons_not_consumed == estmt.num_pcons):
                            if ( estmt.canMarkDead() ):
                                estmt.markDead()
                                # since we mark this dead, we don't have to
                                # enter an eid mapping
                                break
                            else:
                                estmt.markNotConsumed()
                                #
                                # We have to go down and enter the EID found
                                # because this estmt is live
                        else:
                            # We have not reached prod limit. So, we cannot
                            # assume any other prod cons left will be also
                            # dead. For instance, PROD before this arg could
                            # be actually consumed, or the next PROD_CONS may
                            # get consumed. 
                            # Therefore, we have to go to next step
                            # and mark this PROD_CONS as consumed and enter
                            # the eid found for it. However, since multiple
                            # prod_cons are rare, having to enter it would
                            # be the right thing -- i.e., this estmt is 
                            # not going to be dead
                            #
                            pass
                        
                # STEP: Mark consumers and prod_consumers as consumed
                # if there is a consumer (or prod_cons) eid to be marked
                # as consumed, do it now
                #
                if (cons_eid_found != None):

                    estmt.setArgEID(arg, cons_eid_found)    

                    self.markProdAsConsumed(cons_eid_found, blk_path, 
                                            prod_eid_consumed_path)


            # If we failed to link a statement, the whole function would not
            # compile. So, abort and return failure
            #
            if (not stmt_success):
                func_success = False
                break

        # return whether function succeeded
        #
        return func_success


    #------------------------------------------------------------------------
    # PHASE 3 function:
    # Expr objects have parameters that can be mutated (e.g., 'opcode').
    # The following method changes those attributes to produce different
    # code
    #
    def mutateParameters(self):

        func = self.func                # func of the soln

        for estmt in func.estmts:       # for each estmt

            # We don't decide to mutate every stmt
            #
            should_mutate = (random.randint(1,100) <= Knob.stmtMutationOdds)
            #
            if (not should_mutate):     # don't mutate
                continue
            
            estmt.mutateOpcode()        # randomly change the opcode

            if (estmt.isFuncCall()):              # if function call
                
                self.mutateFuncCallExpr(estmt)    # try to change it


    #------------------------------------------------------------------------
    # Mutate a function call expr to another one in the same class
    # We just change the name of the function and its definition string
    # We don't have to change arguments because we pick within the same
    # class. This method does not change any EIDs
    #
    def mutateFuncCallExpr(self, fcall):

        class_id = fcall.soln_class
        solns_of_class = self.all_solns_found[class_id]        

        num_solns = len(solns_of_class)           # num of solution in class

        if (num_solns >= 1):                      # if more than 0 soln

            ind = random.randint(0, num_solns-1)       # pick 1 at random

            new_psol = solns_of_class[ind]             # new soln
            new_fcall = new_psol.sol_func_call_expr    # new fcall
            assert(new_fcall != None)

            # if new name is same as the old name OR 
            # if new name is same as the name of the problem we are solving
            # then just return
            #
            if (new_fcall.func_name == fcall.func_name) or \
               (new_fcall.func_name == self.prob_soln.problem.getName()):
                return

            if (Dbg.evolveOnce > 0):
                print("Mutating ",  fcall.func_name, " to ", 
                      new_fcall.func_name)
                 
            fcall.func_name = new_fcall.func_name       # change name
            fcall.func_def_str = new_fcall.func_def_str # change def str

            assert(fcall.num_args == new_fcall.num_args)
            assert(fcall.arg_types[0] == new_fcall.arg_types[0])
                                

    #------------------------------------------------------------------------
    # Whether the producer path dominates the consumer path
    #
    def canConsume(self, prod_path, cons_path):

        if (Dbg.linkMutate > 0):
            print("canCons: prod:", prod_path, " cons: ", cons_path)

        # if producer is a higher depth, the consumer cannot consume at all
        #
        if (len(cons_path) < len(prod_path)):
            Dbg.prn(Dbg.linkMutate, "\t-False(1)")
            return False

        # See whether the producer levels are subset of consumer levels
        # 
        for ind in range(len(prod_path)):
            if (prod_path[ind] != cons_path[ind]):
                Dbg.prn(Dbg.linkMutate, "\t-False(2)")
                return False

        Dbg.prn(Dbg.linkMutate, "\t-True")
        return True


    #------------------------------------------------------------------------
    # Helper method to update the map of consumed producers
    #
    def markProdAsConsumed(self, prod_eid, cons_path, prod_eid_consumed_path):

        # First get the exiting level at which the producer is consumed
        #
        exist_cons_path = prod_eid_consumed_path.get(prod_eid)   
        #
        if (exist_cons_path == None):               # if no existing consumer

            prod_eid_consumed_path[prod_eid] = cons_path
            
            if (Dbg.linkMutate > 0):
                print("No Exist cons path for prod eid:", prod_eid, 
                      " cons_path:", cons_path)

            return
    
        # we are here because there is an exiting consumer. We have to pick
        # the common part of both paths as the 'consumer path'
        #
        common_path = list()

        for ind in range(len(cons_path)):
        
            if (ind >= len(exist_cons_path)):          
                break
            
            if (cons_path[ind] != exist_cons_path[ind]):
                break
            
            common_path.append(cons_path[ind])

            if (Dbg.linkMutate > 0):
                print("for prod eid:", prod_eid, " cons_path:", cons_path, 
                      " exist:", exist_cons_path, " common:", common_path)

        # set the common path constructed
        #
        prod_eid_consumed_path[prod_eid] = common_path


    #------------------------------------------------------------------------
    # This method does two things to a list of exprs (e.g., an idiom) that
    # is about to be inserted. (1) For producers, it assigns a new EID
    # and records it in 'eids_of_type'. (2) If there is an existing producer/
    # consumer link (through an eid), it preserves that link by renaming the
    # eid of the consumer to the new eid of the producer
    #
    def renameAndAddArgEIDs(self, exprs, start_eid, eids_of_type):

        old2new = dict()                # map from old=>new eid

        for expr in exprs:

            for arg in range(len(expr.arg_types)):     # for all eids

                arg_eid = expr.arg_eids[arg]
                arg_prod_cons = expr.prod_cons[arg]
                arg_type = expr.arg_types[arg]

                if (arg_prod_cons == Expr.PROD):

                    # if there is an existing eid, make a mapping
                    #
                    if (arg_eid != Expr.InvalEID):     # there is valid eid
                        old2new[arg_eid] = start_eid

                    # now, change the value of arg_eid to new one
                    #    
                    expr.arg_eids[arg] = start_eid      # enter new eid
                    new_arg_eid = start_eid
                    start_eid += 1

                    # Add producer eid to eid list of appropriate type
                    #
                    eids_of_type[arg_type].append(new_arg_eid)

                    if (Dbg.renameEID > 0):
                        print("+ prd eid:", new_arg_eid, " for ty:", arg_type) 

                else:
                    # arg is a CONS or PROD_CONS. Usually, consumers don't
                    # have an eid assigned but idioms may have consumers with
                    # assigned eids. So, we have to rename them
                    #
                    if (arg_eid != Expr.InvalEID):     # there is valid eid
                        new_eid = old2new[arg_eid]
                        assert(new_eid != None)        # must have a mapping
                        expr.arg_eids[arg] = new_eid   # enter new eid

        return start_eid


    #------------------------------------------------------------------------
    # Get a producer EID from the eid_list but the producer must be from 
    # upstream cf the req_path.
    # This method can randomly pick a producer, or pick a suitable
    # producer from the end of the list, if so requested
    # Args:
    #   eid_list: list of all producers of the required type
    #   eid2path: map containing the path of each eid producer
    #   req_path: requested level for the consumer
    #   no_random_prod = requests no random selection
    #
    def getEIDFromUpstream(self, eid_list, eid2path, req_path, 
                            no_random_prod):
        #
        if (len(eid_list) == 0):
            return Expr.InvalEID

        # Randomly pick a prod eid from the eid_list to link to
        #
        eid_pos = random.randint(0, len(eid_list)-1)
        eid_picked = eid_list[eid_pos]      # pick a producer

        # if eid_picked is from an incompatible path or 
        # if 'no_random_prod' is True, we traverse backwards 
        # since closer producers to this
        # stmt are in the back of eid_list. We traverse backwards until we
        # see a producer
        #
        is_incorr_level = not self.canConsume(eid2path[eid_picked], req_path)

        if (is_incorr_level or no_random_prod):
            #
            eid_picked = Expr.InvalEID
            #
            if (Dbg.linkMutate > 0):
                print("prods of type:", eid_list)

            for prod_eid in reversed(eid_list):             # go backwards

                if (Dbg.linkMutate > 0):
                    print("prod eid:" , prod_eid, " L:", eid2path[prod_eid])

                if (self.canConsume(eid2path[prod_eid], req_path)):
                    eid_picked = prod_eid                   # pick prod eid
                    break
        
        return eid_picked


    #------------------------------------------------------------------------
    # Get a producer EID from the eid_list. The producer can be from any
    # block level. This method can randomly pick a producer, or pick a
    # producer from the end of the list, if so requested
    #
    def getEIDFromAnyBlk(self, eid_list, no_random_prod):
        #
        if (len(eid_list) == 0):
            return Expr.InvalEID

        # if no_random_prod is specified, pick sequentially
        #
        if (no_random_prod):
            return eid_list[ -1 ]

        else: 
            # Randomly pick a prod eid from the eid_list to link to
            #
            eid_pos = random.randint(0, len(eid_list)-1)
            return eid_list[eid_pos]                   # pick a producer


    #------------------------------------------------------------------------
    # evaluate the function built with arguments and find the return value
    # Note: Conceptually, this function should be part of the Checker base
    #       class but we keep it in this class for simplicity.
    #
    def execFunc(self, prob_name, eid2senior):

        func = self.func                # func of the soln

        if (Dbg.execFunc > 0):
            print("--!!-- Non Renamed Code -!!-")
            print(func.emit(dict()))

        import_str = ""                 # any imports needed
        prestr = ""                     # any preamble to code str
        poststr = ""                    # any postamble to code str

        # Detect infinite loops and terminate they by using signals.
        # Before we launch the code, we set up a signal for 1sec and 
        # setup a signal handler. In case of an infinite loop, the
        # signal goes, and the handler is called, which raises an
        # exception, which is caught by "except" section below. If the
        # func terminates, we cancel the signal
        #
        if (Knob.detectInfiniteLoops):
            importstr = "\nimport signal\n"
            #
            prestr += "\nclass TimeOutEx(Exception):\n\tpass\n"
            prestr += "\ndef handler(s, f):"
            prestr += "\n\traise TimeOutEx\n\n"
            prestr += "signal.signal(signal.SIGALRM, handler)\n"
            prestr += "signal.alarm(1)\n\n"    # alarm in 2 seconds
            #
            poststr = "signal.alarm(0)\n"    # cancel the signal

        # Code string to execute. This has function definitions + 
        # prestr + body
        #
        func_str = func.emit(eid2senior, Expr.EMIT_PLAIN, prestr)

        # Detection recursion. We search the function string, which contains
        # the function tree, to detect any calls to function under 
        # construction. If detected, we abort this function
        #
        if (re.search("\s" + prob_name + "\(", func_str)):
            GlobStat.num_recursions += 1
            if (Dbg.print_recursions):
                print("Recursion detected for prob:", prob_name)
                print(func_str)
            return None

        #import_str = "import __builtin__\n"

        code_str = importstr +  func_str + poststr

        #code_str += "\nprint(return_val)\n"

        if (Dbg.finalCode > 0):
            print("--------- Code -------------")
            print(func_str)

        GlobStat.num_execs += 1

        # global names passed on to exec. Var "return_val" will be set
        # by our code, if successful
        #
        my_globals = globals()

        #my_globals["return_val"] = None     # set to none

        try:
            exec(code_str, my_globals)

        except TimeOutEx:
            signal.alarm(0)                  # disable any pending signal
            GlobStat.num_timeouts += 1       # record stat
            return None

        except KeyboardInterrupt:
            print("At Ctrl-C, Prob:", prob_name, " Code: ---\n", code_str)
            return None

        except Exception:                    # catch all exceptions
            signal.alarm(0)                  # disable any pending signal
            GlobStat.num_excpetions += 1     # increment stat

            if (Dbg.evalException):

                err = sys.exc_info()         # get all exception info 
                err_class = err[0]           # class of exception

                # If there is a NameError
                #
                name_err = type(err_class) is NameError
                syntax_err = type(err_class) is SyntaxError

                if (name_err or syntax_err):
                #if (True):
                    print("Errors caught:", err )
                    print ("Exception occurred. Discarding func:")
                    print("- Code --\n", code_str)

            return None
            

        # Disable any signal still pending
        #
        signal.alarm(0)

        # we are here because code_str executed successfully
        #
        if (Dbg.execFunc > 0):
            print("Output: ", my_globals["return_val"]) 

        # The code produces the return value in global variable "return_val"
        #
        return my_globals["return_val"]


    #------------------------------------------------------------------------
    # This method does two things
    #   (1) Creates a string representation as a callable function,
    #       and records it in prob_soln
    #   (2) Records the function in prob_soln as a solution
    #
    # Inputs:
    #   (1) eid2senior: map for 'phi elimination'
    #   (2) psol: The ProblemSolnPack we solved
    #  
    def packageFunc(self, eid2senior, psol):

        func = self.func                          # get some shortcuts
        prob = psol.problem
        num_args = len(prob.inputs)

        # Remove dead code (clean) and test the cleaned function
        #
        cleaned_func = copy.deepcopy(self.func)
        self.removeDeadExprs(cleaned_func, eid2senior)
        self.testCleanedFunc(cleaned_func, psol, eid2senior)
        #
        self.func_stats.stmt_cnt = len(self.func.estmts)
        self.func_stats.clean_stmt_cnt = len(cleaned_func.estmts)

        # Record the function, its code string, and its eid2senior
        #
        psol.sol_clean_func = cleaned_func             # cleaned func
        psol.sol_func = self.func                      # function built
        psol.sol_eid2senior = eid2senior               # eid mapping
        psol.sol_stats = copy.copy(self.func_stats)    # record stat obj
        psol.is_soln_new = True                        # newly found soln


        if (Dbg.packageFunc > 0):
            print("After clean up len: ", len(psol.sol_clean_func.estmts), 
                  " orig:", len(psol.sol_func.estmts))


        # STEP: Create the code string for this function
        #
        # first, create the function header
        #
        head_str = "\ndef " + prob.getName() + "("  # function header
        
        assert(num_args > 0)                      # add first arg
        head_str += "arg0" 

        for arg in range(1, num_args):            # print other args
            head_str += ", arg" + str(arg) 
            
        head_str += "):\n"                        # close header

        # emit all the estmts in the cleaned function
        #
        code_str = cleaned_func.emit(eid2senior, Expr.EMIT_FUNC, head_str)  
      
        print(" ............. Function Form of Code ................")
        print(code_str)

        # Now create a function call expression for this function and 
        # record it. This function call expr can be inserted into any other
        # function we build, like any other expression. This is one way we
        # build new solutions from solutions found earlier.
        #
        prob = psol.problem                       # shortcut
        #
        assert(len(prob.output_types) == 1)       # TODO: support tuple return
        ret_type = prob.output_types[0]           # return type
        #
        fcall_expr = \
            FuncCallExpr(prob.getName(), ret_type, prob.input_types, 
                         code_str, prob.prod_cons_arg)

        psol.sol_func_call_expr = fcall_expr      # record func call expr

        # Record the solution class of this function call
        #
        psol.sol_func_call_expr.soln_class = self.cur_soln_class

        # Now, add function call expression to expression list, so that, it
        # can be used in future evolution
        #
        if Knob.addDiscovedCallsWithinRank:
            self.ei_store.addFuncCallExpr(fcall_expr)

        # Add this psol to solutions found list -- so that we can evolve it
        # for new problems. Also, add a cleaned up version
        #
        self.solns_found.append(psol)

        # Debug print 
        #
        fcall_str = fcall_expr.emit()
        Dbg.prn(Dbg.packageFunc, "\nFunc Call: " + fcall_str)

        psol.sol_stats.prob_name = psol.problem.getName()
        psol.sol_stats.sol_time = getTimeStamp()

        psol.sol_stats.dump()                     # print stats for soln


################## ProblemGen/Checkers: Max & Min ###########################

# This section shows how to create ProblemGenerators and Checkers by deriving
# them from the base classes and implementing generateNewInputs() and check()
# methods


#------------------------------------------------------------------------------
# Problem generator for "max" algo. Just generates an input array. It can be
# used by similar problems (like min), which accept same inputs/outputs.
#
class MaxProbGenerator(ProblemGenerator):
        def __init__(self, problem):
            super().__init__(problem)

            # Set input and output types 
            # This problem has one input array
            #
            self.prob.addInputType(Expr.ARR)      # add input type
            self.prob.addOutputType(Expr.ELEM)    # add output type


        # Method for generating a new input for the problem
        #
        def generateNewInputs(self):
            #
            inarr = []                            # new input arr
            #
            while (len(inarr) < self.prob.prob_size):   # init input arr
                num = random.randint(-25, 25)
                inarr.append(num)

            # Special test case:
            # For one problem size, fill the entire array with zeros
            #
            if (self.prob.prob_size == (self.prob.def_prob_size + 1)):
                inarr = [0] * self.prob.prob_size
            
            # Set the new input
            #
            self.prob.setInput(0, inarr)          # set actual input


#------------------------------------------------------------------------------
# Checker for Max
#
class MaxChecker(Checker):
    def __init__(self, problem):
        super().__init__(problem)

    # Checking routine for max. Goes thru each element and checks whether
    # any element in input is greater than the output
    #
    def check(self):
        inarr = self.prob.inputs[0]
        out = self.prob.outputs[0]
        seen = 0;

        for x in inarr:
            if x > out:                 # input arr has an elem > output 
                return False
            if x == out:                # saw the output elem in input arr
                seen = 1

        if (seen == 0):                 # output value not found in input
            Dbg.prn(Dbg.checker, '\t-check failed: value not found');
            return False

        return True



#------------------------------------------------------------------------------
# Checker for MaxIndex
#
class MaxIndexChecker(Checker):
    def __init__(self, problem):
        super().__init__(problem)

    # Checking routine for max. Goes thru each element and checks whether
    # any element in input is greater than the output
    #
    def check(self):
        inarr = self.prob.inputs[0]
        outind = self.prob.outputs[0]
        seen = 0;

        if (not isinstance(outind, int)): # check for int index
            return False

        # if index out of range
        #
        if ((outind < 0) or (outind >= len(inarr))):     
            return False

        out = inarr[outind]             # get value at index

        for x in inarr:
            if x > out:                 # input arr has an elem > output 
                return False

        return True


#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# Checker for Min
#
class MinChecker(Checker):
    def __init__(self, problem):
        super().__init__(problem)

    # Checking routine for max. Goes thru each element and checks whether
    # any element in input is greater than the output
    #
    def check(self):
        inarr = self.prob.inputs[0]
        out = self.prob.outputs[0]
        seen = 0;

        for x in inarr:
            if x < out:                 # input arr has an elem > output 
                return False
            if x == out:                # saw the output elem in input arr
                seen = 1

        if (seen == 0):                 # output value not found in input
            Dbg.prn(Dbg.checker, '\t-check failed: value not found');
            return False

        return True


#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# Checker for Arr Len
#
class ArrLenChecker(Checker):
    def __init__(self, problem):
        super().__init__(problem)
    #
    def check(self):
        inarr = self.prob.inputs[0]
        out = self.prob.outputs[0]

        return (out == len(inarr))

#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# Checker for Arr Sum
#
class ArrSumChecker(Checker):
    def __init__(self, problem):
        super().__init__(problem)
    #
    def check(self):
        inarr = self.prob.inputs[0]
        out = self.prob.outputs[0]

        arrsum = 0
        for x in inarr:
            arrsum += x

        return (out == arrsum)


#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# Checker for Sum Of Squares
#
class SumOfSquaresChecker(Checker):
    def __init__(self, problem):
        super().__init__(problem)
    #
    def check(self):
        inarr = self.prob.inputs[0]
        out = self.prob.outputs[0]

        arrsum = 0
        for x in inarr:
            arrsum += x * x

        return (out == arrsum)



#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# Checker for Arr Avg (rounded down)
#
class AvgChecker(Checker):
    def __init__(self, problem):
        super().__init__(problem)
    #
    def check(self):
        inarr = self.prob.inputs[0]
        out = self.prob.outputs[0]

        arrsum = 0
        for x in inarr:
            arrsum += x

        return (out == (arrsum // len(inarr)))


#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# Checker for Arr Avg If LT (rounded down)
#
class ScaledAvgChecker(Checker):
    def __init__(self, problem):
        super().__init__(problem)
    #
    def check(self):
        inarr = self.prob.inputs[0]
        inelem = self.prob.inputs[1]
        out = self.prob.outputs[0]

        arrsum = 0
        for x in inarr:
            arrsum += x * inelem

        return (out == (arrsum // len(inarr)))


#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# Sort problem Generator
#
class SortProbGenerator(ProblemGenerator):
        def __init__(self, problem):
            super().__init__(problem)

            # Set input and output types 
            # This problem has one input array
            #
            self.prob.addInputType(Expr.ARR)      # add input type
            self.prob.addOutputType(Expr.ARR)     # add output type


        # Method for generating a new input for the problem
        #
        def generateNewInputs(self):
            #
            inarr = []                            # new input arr
            #
            while (len(inarr) < self.prob.prob_size):   # init input arr
                num = random.randint(-100, 100)
                inarr.append(num)
            
            # Set the new input
            #
            self.prob.setInput(0, inarr)          # set actual input


#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# Checker for Sort
#
class SortCheckerDescend(Checker):
    def __init__(self, problem):
        super().__init__(problem)

    # Checking routine for max. Goes thru each element and checks whether
    # any element in input is greater than the output
    #
    def check(self):
        inarr = self.prob.inputs[0]
        outarr = self.prob.outputs[0]

        if (len(inarr) != len(outarr)):           # check if lengths are equal
            return False

        sort_arr = inarr.copy()                   # create a copy and sort
        sort_arr.sort(reverse=True)               # sort descending
        
        for ind in range(len(sort_arr)):
            if (sort_arr[ind] != outarr[ind]):
                return False

        return True



#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#
class SortCheckerAscend(Checker):
    def __init__(self, problem):
        super().__init__(problem)


    # Checking routine for max. Goes thru each element and checks whether
    # any element in input is greater than the output
    #
    def check(self):
        inarr = self.prob.inputs[0]
        outarr = self.prob.outputs[0]

        if (len(inarr) != len(outarr)):           # check if lengths are equal
            return False

        sort_arr = inarr.copy()                   # create a copy and sort
        sort_arr.sort()                           # sort ascending
        
        for ind in range(len(sort_arr)):
            if (sort_arr[ind] != outarr[ind]):
                return False

        return True


#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#
class reverseChecker(Checker):
    def __init__(self, problem):
        super().__init__(problem)

    # Checking routine for max. Goes thru each element and checks whether
    # any element in input is greater than the output
    #
    def check(self):
        inarr = self.prob.inputs[0]
        outarr = self.prob.outputs[0]

        if (len(inarr) != len(outarr)):           # check if lengths are equal
            return False

        rev_arr = inarr.copy()                    # make a copy and reverse
        rev_arr.reverse()

        for ind in range(len(inarr)):              # sum of output arr
            if ( outarr[ind] != rev_arr[ind] ):
                return False
            
        return True

#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# Add 2 arrays together
#
class ArrAddProbGenerator(ProblemGenerator):
        def __init__(self, problem):
            super().__init__(problem)

            # Set input and output types 
            # This problem has one input array
            #
            self.prob.addInputType(Expr.ARR)      # add input type
            self.prob.addInputType(Expr.ARR)      # add input type
            self.prob.addOutputType(Expr.ARR)     # add output type


        # Method for generating a new input for the problem
        #
        def generateNewInputs(self):
            #
            inarr1 = []                            # new input arr
            inarr2 = []                            # new input arr
            #
            while (len(inarr1) < self.prob.prob_size):   # init input arr
                num = random.randint(-100, 100)
                inarr1.append(num)
                num = random.randint(-100, 100)
                inarr2.append(num)
            
            # Set the new input
            #
            self.prob.setInput(0, inarr1)          # set actual input
            self.prob.setInput(1, inarr2)          # set actual input





#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# Add 2 arrays together
#
class ArrAvgProbGenerator(ProblemGenerator):
        def __init__(self, problem):
            super().__init__(problem)

            # Set input and output types 
            # This problem has one input array
            #
            self.prob.addInputType(Expr.ARR)      # add input type
            self.prob.addOutputType(Expr.ELEM)     # add output type


        # Method for generating a new input for the problem
        #
        def generateNewInputs(self):
            #
            inarr1 = []                            # new input arr
            #
            while (len(inarr1) < self.prob.prob_size):   # init input arr
                num = random.randint(-10, 100)
                inarr1.append(num)
            
            # Set the new input
            #
            self.prob.setInput(0, inarr1)          # set actual input


#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#
class ArrAddChecker(Checker):
    def __init__(self, problem):
        super().__init__(problem)

    # Checking routine for max. Goes thru each element and checks whether
    # any element in input is greater than the output
    #
    def check(self):
        inarr1 = self.prob.inputs[0]
        inarr2 = self.prob.inputs[1]
        outarr = self.prob.outputs[0]

        if (len(inarr1) != len(outarr)):           # check if lengths are equal
            return False

        assert(len(inarr1) == len(inarr2))

        #print("ArrAddChecker In1:", inarr1, " in2:", inarr2, " Out:", outarr)

        for ind in range(len(inarr1)):
            if (outarr[ind] != (inarr1[ind] + inarr2[ind])):
                return False

        return True


#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# Add 2 arrays together
#
class MatVecProbGenerator(ProblemGenerator):
        def __init__(self, problem):
            super().__init__(problem)

            # Set input and output types 
            # This problem has one input array
            #
            self.prob.addInputType(Expr.ARR_OF_ARR)    # add input type
            self.prob.addInputType(Expr.ARR)           # add input type
            self.prob.addOutputType(Expr.ARR)          # add output type

        # Method for generating a new input for the problem
        #
        def generateNewInputs(self):
            #
            matrix = []                          # new input arr
            vec = []                             # new input arr
            #
            probsize = self.prob.prob_size
            num_rows = random.randint(1, 5)      # pick a random # of rows

            for col in range(probsize):          # generate input vec
                num = random.randint(-100, 100)
                vec.append(num)

            for row in range(num_rows):          # for each row of matrix

                row_arr = list()

                for col in range(probsize):
                    num = random.randint(-100, 100)
                    row_arr.append(num)
                
                matrix.append(row_arr)           # add row to matrix
                
                assert(len(matrix[row]) == len(vec))

            
            # Set the new input
            #
            self.prob.setInput(0, matrix)         # set actual input
            self.prob.setInput(1, vec)            # set actual input

            if (Dbg.checker > 0):
                print("Gen: Mat:", matrix, " vec:", vec)


#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#
class MatVecChecker(Checker):
    def __init__(self, problem):
        super().__init__(problem)

    # Checking routine for max. Goes thru each element and checks whether
    # any element in input is greater than the output
    #
    def check(self):
        matrix = self.prob.inputs[0]
        vec = self.prob.inputs[1]
        outarr = self.prob.outputs[0]

        if (Dbg.checker > 0):
            rint("Check: Mat:", matrix, " vec:", vec, " out:", outarr)

        if (len(matrix) != len(outarr)):          # check if lengths are equal
            return False

        ncols = len(vec)
        nrows = len(matrix)

        for row in range(nrows):                  # for each row in matrix

            row_vec = matrix[row]
            dotp = 0

            assert(len(row_vec) == len(vec))

            for col in range(ncols):              # for each col in row

                dotp += row_vec[col] * vec[col]

            if (outarr[row] != dotp):
                return False

        return True



#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#
class MatAddVecChecker(Checker):
    def __init__(self, problem):
        super().__init__(problem)

    # Checking routine for max. Goes thru each element and checks whether
    # any element in input is greater than the output
    #
    def check(self):
        matrix = self.prob.inputs[0]
        vec = self.prob.inputs[1]
        outarr = self.prob.outputs[0]

        if (Dbg.checker > 0):
            rint("Check: Mat:", matrix, " vec:", vec, " out:", outarr)

        if (len(matrix) != len(outarr)):          # check if lengths are equal
            return False

        ncols = len(vec)
        nrows = len(matrix)

        for row in range(nrows):                  # for each row in matrix

            row_vec = matrix[row]
            dotp = 0

            assert(len(row_vec) == len(vec))

            for col in range(ncols):              # for each col in row

                dotp += row_vec[col] + vec[col]

            if (outarr[row] != dotp):
                return False

        return True



#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#
class ArrMultChecker(Checker):
    def __init__(self, problem):
        super().__init__(problem)

    # Checking routine for max. Goes thru each element and checks whether
    # any element in input is greater than the output
    #
    def check(self):
        inarr1 = self.prob.inputs[0]
        inarr2 = self.prob.inputs[1]
        outarr = self.prob.outputs[0]

        if (len(inarr1) != len(outarr)):           # check if lengths are equal
            return False

        assert(len(inarr1) == len(inarr2))

        #print("ArrAddChecker In1:", inarr1, " in2:", inarr2, " Out:", outarr)

        for ind in range(len(inarr1)):
            if (outarr[ind] != (inarr1[ind] * inarr2[ind])):
                return False

        return True



#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# Add 2 arrays together
#
class DotProdProbGenerator(ProblemGenerator):
        def __init__(self, problem):
            super().__init__(problem)

            # Set input and output types 
            # This problem has one input array
            #
            self.prob.addInputType(Expr.ARR)      # add input type
            self.prob.addInputType(Expr.ARR)      # add input type
            self.prob.addOutputType(Expr.ELEM)     # add output type


        # Method for generating a new input for the problem
        #
        def generateNewInputs(self):
            #
            inarr1 = []                            # new input arr
            inarr2 = []                            # new input arr
            #
            while (len(inarr1) < self.prob.prob_size):   # init input arr
                num = random.randint(-100, 100)
                inarr1.append(num)
                num = random.randint(-100, 100)
                inarr2.append(num)
            
            # Set the new input
            #
            self.prob.setInput(0, inarr1)          # set actual input
            self.prob.setInput(1, inarr2)          # set actual input


#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#
class DotProductChecker(Checker):
    def __init__(self, problem):
        super().__init__(problem)

    # Checking routine for max. Goes thru each element and checks whether
    # any element in input is greater than the output
    #
    def check(self):
        inarr1 = self.prob.inputs[0]
        inarr2 = self.prob.inputs[1]
        out = self.prob.outputs[0]

        assert(len(inarr1) == len(inarr2))

        dotp = 0
        for ind in range(len(inarr1)):
            dotp += inarr1[ind] * inarr2[ind]

        return (dotp == out)


#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#
class SumAddChecker(Checker):
    def __init__(self, problem):
        super().__init__(problem)

    #
    def check(self):
        inarr1 = self.prob.inputs[0]
        inarr2 = self.prob.inputs[1]
        out = self.prob.outputs[0]

        assert(len(inarr1) == len(inarr2))

        dotp = 0
        for ind in range(len(inarr1)):
            dotp += inarr1[ind] + inarr2[ind]

        return (dotp == out)




#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# Add 2 arrays together
#
class AddElemProbGenerator(ProblemGenerator):
        def __init__(self, problem):
            super().__init__(problem)

            # Set input and output types 
            # This problem has one input array
            #
            self.prob.addInputType(Expr.ARR)      # add input type
            self.prob.addInputType(Expr.ELEM)      # add input type
            self.prob.addOutputType(Expr.ARR)     # add output type


        # Method for generating a new input for the problem
        #
        def generateNewInputs(self):
            #
            inarr = []                            # new input arr
            inelem = 0                            # new input arr
            #
            while (len(inarr) < self.prob.prob_size):   # init input arr
                num = random.randint(-100, 100)
                inarr.append(num)

            if (num > 90):                        # use zero 5% of the time
                inelem = 0
            else:
                inelem = random.randint(-100, 100)


            
            
            # Set the new input
            #
            self.prob.setInput(0, inarr)           # set actual input
            self.prob.setInput(1, inelem)          # set actual input


#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# Is given element in the array
#
class IsInArrProbGenerator(ProblemGenerator):
        def __init__(self, problem):
            super().__init__(problem)

            # Set input and output types 
            # This problem has one input array
            #
            self.prob.addInputType(Expr.ARR)       # add input type
            self.prob.addInputType(Expr.ELEM)      # add input type
            self.prob.addOutputType(Expr.BOOL)     # add output type


        # Method for generating a new input for the problem
        #
        def generateNewInputs(self):
            #
            inarr = []                            # new input arr
            inelem = 0                            # new input element
            bound = 100
            #
            while (len(inarr) < self.prob.prob_size):   # init input arr
                num = random.randint(-bound, bound)
                inarr.append(num)

            # If we just pick a random number as inelem, most of the time
            # it will be not in inarr. That is not a good checker. So, we
            # try to pick an elem from array about 25% of the time
            #
            pos = random.randint(0, self.prob.prob_size-1)
            r = random.randint(0,99)
            should_pick = (r <= 25)
            if (should_pick):
                inelem = inarr[pos]     # pick a number within the array
            elif r <= 30:
                inelem = 0              # use 0 about 5% of the time
            else:
                # Generate a number
                inelem = random.randint(-2*bound, 2*bound)

            # Set the new input
            #
            self.prob.setInput(0, inarr)           # set actual input
            self.prob.setInput(1, inelem)          # set actual input




#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#
class AddElemChecker(Checker):
    def __init__(self, problem):
        super().__init__(problem)

    # Checking routine for max. Goes thru each element and checks whether
    # any element in input is greater than the output
    #
    def check(self):
        inarr = self.prob.inputs[0]
        inelem = self.prob.inputs[1]
        outarr = self.prob.outputs[0]

        # print("DescChecker In:", inarr, " Out:", outarr)

        if (len(inarr) != len(outarr)):           # check if lengths are equal
            return False

        for ind in range(len(inarr)):
            if (outarr[ind] != (inarr[ind] + inelem)):
                return False

        return True


#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#
class SubElemChecker(Checker):
    def __init__(self, problem):
        super().__init__(problem)

    # Checking routine for max. Goes thru each element and checks whether
    # any element in input is greater than the output
    #
    def check(self):
        inarr = self.prob.inputs[0]
        inelem = self.prob.inputs[1]
        outarr = self.prob.outputs[0]

        # print("DescChecker In:", inarr, " Out:", outarr)

        if (len(inarr) != len(outarr)):           # check if lengths are equal
            return False

        for ind in range(len(inarr)):
            if (outarr[ind] != (inarr[ind] - inelem)):
                return False

        return True




#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#
class ScaleArrChecker(Checker):
    def __init__(self, problem):
        super().__init__(problem)

    # Checking routine for max. Goes thru each element and checks whether
    # any element in input is greater than the output
    #
    def check(self):
        inarr = self.prob.inputs[0]
        inelem = self.prob.inputs[1]
        outarr = self.prob.outputs[0]

        # print("DescChecker In:", inarr, " Out:", outarr)

        if (len(inarr) != len(outarr)):           # check if lengths are equal
            return False

        for ind in range(len(inarr)):
            if (outarr[ind] != (inarr[ind] * inelem)):
                return False

        return True


#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#
class IsInArrChecker(Checker):
    def __init__(self, problem):
        super().__init__(problem)

    # Checking routine for max. Goes thru each element and checks whether
    # any element in input is greater than the output
    #
    def check(self):
        inarr = self.prob.inputs[0]
        inelem = self.prob.inputs[1]
        out = self.prob.outputs[0]

        found = False
        for x in inarr:
            if (x == inelem):
                found = True

        return (out == found)



#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# Prob generator for Index Of
#
class IndexOfProbGenerator(ProblemGenerator):
        def __init__(self, problem):
            super().__init__(problem)

            # Set input and output types
            # This problem has one input array
            #
            self.prob.addInputType(Expr.ARR)       # add input type
            self.prob.addInputType(Expr.ELEM)      # add input type
            self.prob.addOutputType(Expr.ELEM)     # add output type


        # Method for generating a new input for the problem
        #
        def generateNewInputs(self):
            #
            inarr = []                            # new input arr
            inelem = 0                            # new input element
            #
            while (len(inarr) < self.prob.prob_size):   # init input arr
                num = random.randint(-100, 100)
                inarr.append(num)

            pos = random.randint(0, self.prob.prob_size-1)
            inelem = inarr[pos]     # pick a number within the array

            # Set the new input
            #
            self.prob.setInput(0, inarr)           # set actual input
            self.prob.setInput(1, inelem)          # set actual input


#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# Prob generator for Remove
# CAUTION: Remove has a PROD_CONS arg as its 1st argument
#
class RemoveProbGenerator(ProblemGenerator):
        def __init__(self, problem):
            super().__init__(problem)

            # Set input and output types
            # This problem has one input array
            #
            self.prob.addInputType(Expr.ARR)       # add input type
            self.prob.addInputType(Expr.ELEM)      # add input type
            self.prob.addOutputType(Expr.ARR)     # add output type

            # Set the 0th arg as a PROD_CONS
            # For remove, 0th arg is both a consumer and a producer
            # E.g., in Python: arr1.remove(val) -- arr1 is PROD_CONS
            #
            self.prob.setArgAsProdCons(0)


        # Method for generating a new input for the problem
        #
        def generateNewInputs(self):
            #
            inarr = []                            # new input arr
            inelem = 0                            # new input element
            #
            while (len(inarr) < self.prob.prob_size):   # init input arr
                num = random.randint(-100, 100)
                inarr.append(num)

            pos = random.randint(0, self.prob.prob_size-1)
            inelem = inarr[pos]     # pick a number within the array

            # Set the new input
            #
            self.prob.setInput(0, inarr)           # set actual input
            self.prob.setInput(1, inelem)          # set actual input




#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# Prob generator for CntEQ
#
class CntEQProbGenerator(ProblemGenerator):
        def __init__(self, problem):
            super().__init__(problem)

            # Set input and output types
            # This problem has one input array
            #
            self.prob.addInputType(Expr.ARR)       # add input type
            self.prob.addInputType(Expr.ELEM)      # add input type
            self.prob.addOutputType(Expr.ELEM)     # add output type


        # Method for generating a new input for the problem
        #
        def generateNewInputs(self):
            #
            inarr = []                            # new input arr
            inelem = 0                            # new input element
            #
            while (len(inarr) < self.prob.prob_size):   # init input arr
                num = random.randint(-100, 100)
                inarr.append(num)

            # If we just pick a random number as inelem, most of the time
            # it will be not in inarr. That is not a good checker. So, we
            # try to pick an elem from array about 25% of the time. Also,
            # about 5% of the time, use 0 as the element to 
            #
            pos = random.randint(0, self.prob.prob_size-1)
            r = random.randint(0,99)
            should_pick = (r <= 25)
            if (should_pick):
                inelem = inarr[pos]     # pick a number within the array
            elif r <= 30:               # about 5% of the time, use 0      
                inelem = 0              # provide 0s as well for checking
            else:
                inelem = random.randint(-200,200)  # twice the range

            # Set the new input
            #
            self.prob.setInput(0, inarr)           # set actual input
            self.prob.setInput(1, inelem)          # set actual input


#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# Count equal checker
#
class CntEQChecker(Checker):
    def __init__(self, problem):
        super().__init__(problem)

    # Checking routine for max. Goes thru each element and checks whether
    # any element in input is greater than the output
    #
    def check(self):
        inarr = self.prob.inputs[0]
        inelem = self.prob.inputs[1]
        out = self.prob.outputs[0]

        return (inarr.count(inelem) == out)



#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#
class LastIndexOfChecker(Checker):
    def __init__(self, problem):
        super().__init__(problem)

    # Checking routine for max. Goes thru each element and checks whether
    # any element in input is greater than the output
    #
    def check(self):
        inarr = self.prob.inputs[0]
        inelem = self.prob.inputs[1]
        outind = self.prob.outputs[0]

        if (not isinstance(outind, int)): # check for int index
            return False

        # if index out of range
        #
        if ((outind < 0) or (outind >= len(inarr))):
            return False

        index = -1

        for ind, elem in enumerate(inarr):
            if elem == inelem:
                index = ind            

        return (index == outind)


#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# PRE: Element must exist in the list to remove
#      If multiple elements presents, removers last element
#
class RemoveLastChecker(Checker):
    def __init__(self, problem):
        super().__init__(problem)

    # Checking routine for max. Goes thru each element and checks whether
    # any element in input is greater than the output
    #
    def check(self):
        inarr = self.prob.inputs[0]
        inelem = self.prob.inputs[1]
        outarr = self.prob.outputs[0]

        # out array must be missing one element
        #
        if ((len(outarr) + 1) != len(inarr)):
            return False;

        index = -1

        # find index of last element
        #
        for ind, elem in enumerate(inarr):
            if elem == inelem:
                index = ind            

        if (index < 0):          # value not found
            return False

        incopy = inarr.copy()
        incopy.pop(index)
        
        assert(len(incopy) == len(outarr))

        for ind in range(len(incopy)):
            if incopy[ind] != outarr[ind]:
                return False

        return True



#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# PRE: Element must exist in the list to remove
#      If multiple elements presents, removers first element
#
class RemoveFirstChecker(Checker):
    def __init__(self, problem):
        super().__init__(problem)

    # Checking routine for max. Goes thru each element and checks whether
    # any element in input is greater than the output
    #
    def check(self):
        inarr = self.prob.inputs[0]
        inelem = self.prob.inputs[1]
        outarr = self.prob.outputs[0]

        # outarray must be missing one element
        #
        if ((len(outarr) + 1) != len(inarr)):
            return False;

        index = -1

        # find index of last element
        #
        for ind, elem in enumerate(inarr):
            if elem == inelem:
                index = ind
                break

        if (index < 0):          # value not found
            return False

        incopy = inarr.copy()
        incopy.pop(index)
        
        assert(len(incopy) == len(outarr))

        for ind in range(len(incopy)):
            if incopy[ind] != outarr[ind]:
                return False

        return True



#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#
class FirstIndexOfChecker(Checker):
    def __init__(self, problem):
        super().__init__(problem)

    # Checking routine for max. Goes thru each element and checks whether
    # any element in input is greater than the output
    #
    def check(self):
        inarr = self.prob.inputs[0]
        inelem = self.prob.inputs[1]
        outind = self.prob.outputs[0]

        if (not isinstance(outind, int)): # check for int index
            return False

        # if index out of range
        #
        if ((outind < 0) or (outind >= len(inarr))):
            return False

        index = -1

        for ind, elem in enumerate(inarr):
            if elem == inelem:
                index = ind            
                break

        return (index == outind)


#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# Count Greater Than checker
#
class CntGTChecker(Checker):
    def __init__(self, problem):
        super().__init__(problem)

    # Checking routine for max. Goes thru each element and checks whether
    # any element in input is greater than the output
    #
    def check(self):
        inarr = self.prob.inputs[0]
        inelem = self.prob.inputs[1]
        out = self.prob.outputs[0]

        cnt = 0
        for x in inarr:
            if x > inelem:
                cnt += 1

        return (cnt == out)


#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# Count Less Than checker
#
class CntLTChecker(Checker):
    def __init__(self, problem):
        super().__init__(problem)

    # Checking routine for max. Goes thru each element and checks whether
    # any element in input is greater than the output
    #
    def check(self):
        inarr = self.prob.inputs[0]
        inelem = self.prob.inputs[1]
        out = self.prob.outputs[0]

        cnt = 0
        for x in inarr:
            if x < inelem:
                cnt += 1

        return (cnt == out)



#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# Sum if less than checker
#
class SumIfLTChecker(Checker):
    def __init__(self, problem):
        super().__init__(problem)

    # Checking routine for max. Goes thru each element and checks whether
    # any element in input is greater than the output
    #
    def check(self):
        inarr = self.prob.inputs[0]
        inelem = self.prob.inputs[1]
        out = self.prob.outputs[0]

        mysum = 0
        for x in inarr:
            if x < inelem:
                mysum += x

        return (mysum == out)


#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# Sum if less than checker
#
class SumIfGTChecker(Checker):
    def __init__(self, problem):
        super().__init__(problem)

    # Checking routine for max. Goes thru each element and checks whether
    # any element in input is greater than the output
    #
    def check(self):
        inarr = self.prob.inputs[0]
        inelem = self.prob.inputs[1]
        out = self.prob.outputs[0]

        mysum = 0
        for x in inarr:
            if x > inelem:
                mysum += x

        return (mysum == out)



#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# Sum if less than checker
#
class SumIfEQChecker(Checker):
    def __init__(self, problem):
        super().__init__(problem)

    # Checking routine for max. Goes thru each element and checks whether
    # any element in input is greater than the output
    #
    def check(self):
        inarr = self.prob.inputs[0]
        inelem = self.prob.inputs[1]
        out = self.prob.outputs[0]

        mysum = 0
        for x in inarr:
            if x == inelem:
                mysum += x

        return (mysum == out)


#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# Scaled sum
#
class ScaledSumChecker(Checker):
    def __init__(self, problem):
        super().__init__(problem)

    # Checking routine for max. Goes thru each element and checks whether
    # any element in input is greater than the output
    #
    def check(self):
        inarr = self.prob.inputs[0]
        inelem = self.prob.inputs[1]
        out = self.prob.outputs[0]

        mysum = 0
        for x in inarr:
                mysum += (x * inelem)

        return (mysum == out)





#############################################################################
# Class capturing a Framework (one per Process)
#
class Framework:

    def __init__(self, procid, epoch):

        # An Object of ProbSolnList containing all problems/solutions.
        # This is set later
        #
        self.all_psol_class_list = None

        self.eiStore = ExprIdiomStore() # All idioms and expressions

        # TODO: Not used yet
        #
        self.all_evolvers = list()      # different evolution algorithms

        self.procid = procid            # for record keeping
        self.epoch = epoch              # for record keeping


    #------------------------------------------------------------------------
    # Main entry point into evolver
    #
    def evolve(self, stmt_limit):

        random.seed()                   # seed the random num generator

        # Create an evolution algorithm
        # Note: We can have several evolution algorithms in the future
        #
        evolver = Evolver(self.all_psol_class_list, self.eiStore, self.epoch)

        # Set evolver properties used for stats
        #
        evolver.func_stats.sol_procid = self.procid
        evolver.func_stats.sol_epoch = self.epoch

        # Now, do the actual evolution and try to find a solution
        #
        evolver.evolveNSteps(Knob.num_steps_per_epoch, stmt_limit)





        
#############################################################################
# Class capturing all problem/solutions we have to solve
# Note that 'all_prob_solns' is a *list of lists*. All problem/solutions
# that belong to a given 'Class' are in a single list. Therefore, there is
# a list for each class. All such 'class lists' constitute the 'all_prob_solns'
# list.
# All problem/solutions in the same class have the same input args -- i.e., 
# the same number and types -- and the same return types
#############################################################################
class AllProbSolnPack:

    def __init__(self):

        self.all_psol_classes = list()    # All problem_solution units

        # The following describes problem classes. A problem should be
        # added to one of these classes
        #
        self.arr2elem_class = list()    # Array input, single element out
        self.arr2arr_class = list()     # Array input, arr out
        self.arrelem2arr_class = list() # Arr + Elem input, arr output
        self.arrelem2bool_class = list()# Arr + Elem input, bool output
        self.arrelem2elem_class = list()# Arr + Elem input, bool output
        self.twoarr2arr_class = list()  # 2 Arrays of input, arr out
        self.twoarr2elem_class = list()  # 2 Arrays of input, arr out
        self.aOa_arr2arr_class = list() # arr_of_arrays, arr ==> array
        

        if (Knob.solve_groupA or (Knob.group2solve == 1)):
            self.setupGroupAProblems()
        
        if (Knob.solve_groupB or (Knob.group2solve == 2)):
            self.setupGroupBProblems()
        
        if (Knob.solve_groupC or (Knob.group2solve == 3)):
            self.setupGroupCProblems()


        # These are only for testing purposes
        #
        #self.setupSortProblems()
        #
        #self.test()


    #------------------------------------------------------------------------
    # Setup all list problems
    #
    def setupGroupAProblems(self):

        # First, add classes to all problem list
        #
        self.all_psol_classes.append(self.arr2elem_class)
        self.all_psol_classes.append(self.arr2arr_class)
        #
        self.all_psol_classes.append(self.arrelem2arr_class)

        self.all_psol_classes.append(self.arrelem2elem_class)

        # Now, setup
        #
        self.setupMax()
        self.setupMin()
        self.setupSortDescend()
        self.setupSortAscend()
        #self.setupArrLen()
        self.setupReverse()

        # uncomment this if list problems are not run
        # self.all_psol_classes.append(self.arr2elem_class)

        self.setupLastIndexOf()
        self.setupFirstIndexOf()

        self.setupRemoveLast()
        self.setupRemoveFirst()

        self.all_psol_classes.append(self.arrelem2bool_class)
        self.setupIsInArr()
    
    #------------------------------------------------------------------------
    # Sets up vector/matrix problems
    #
    def setupGroupBProblems(self):

        self.all_psol_classes.append(self.twoarr2arr_class)
        self.setupArrAdd()
        self.setupArrMult()

        self.all_psol_classes.append(self.arr2elem_class)
        self.setupSum()
        #self.setupArrLen()

        self.all_psol_classes.append(self.twoarr2elem_class)
        self.setupDotProduct()

        self.all_psol_classes.append(self.aOa_arr2arr_class)
        self.setupMatVec()

        self.all_psol_classes.append(self.arrelem2arr_class)
        self.setupAddElem()
        self.setupSubElem()
        self.setupScaleArr()

        self.all_psol_classes.append(self.arrelem2elem_class)
        self.setupScaledSum()
        self.setupSumOfSquares()


    #------------------------------------------------------------------------
    #
    def setupGroupCProblems(self):

        self.all_psol_classes.append(self.arrelem2elem_class)
        self.setupCntEQ()
        self.setupCntLT()
        self.setupCntGT()
    
        self.setupSumIfLT()
        self.setupSumIfGT()
        self.setupSumIfEQ()
    
        self.all_psol_classes.append(self.arr2elem_class)
        self.setupSum()
        self.setupArrLen()

        self.setupAvg()
        self.setupScaledAvg()


    #------------------------------------------------------------------------
    # Only ingredients for sort -- must be run alone
    #
    def setupSortProblems(self):


        self.all_psol_classes.append(self.arr2elem_class) 
        self.all_psol_classes.append(self.arr2arr_class) 
        self.all_psol_classes.append(self.arrelem2elem_class) 
        self.all_psol_classes.append(self.arrelem2arr_class) 

        self.setupLastIndexOf()
        self.setupRemoveLast()

        self.setupMax()
        self.setupMin()

        self.setupSortDescend()
        self.setupSortAscend()


    #------------------------------------------------------------------------
    #
    def test(self):
        self.all_psol_classes.append(self.arr2elem_class)
        self.setupSum()
        self.setupArrLen()
        self.setupAvg()



    #------------------------------------------------------------------------
    #
    def setupCntEQ(self):

        prob = Problem("CountEQ", 5, 1, Knob.max_probsz)  

        prob_gen = CntEQProbGenerator(prob)  # same probgen as count EQ
        checker = CntEQChecker(prob)

        prob_sol = ProbSolnPack(prob_gen, checker)
        self.arrelem2elem_class.append(prob_sol)



    #------------------------------------------------------------------------
    #
    def setupCntLT(self):

        prob = Problem("CountLT", 5, 1, Knob.max_probsz)  

        prob_gen = CntEQProbGenerator(prob)  # same probgen as count EQ
        checker = CntLTChecker(prob)

        prob_sol = ProbSolnPack(prob_gen, checker)
        self.arrelem2elem_class.append(prob_sol)


    #------------------------------------------------------------------------
    #
    def setupCntGT(self):

        prob = Problem("CountGT", 5, 1, Knob.max_probsz)  

        prob_gen = CntEQProbGenerator(prob)  # same probgen as count EQ
        checker = CntGTChecker(prob)

        prob_sol = ProbSolnPack(prob_gen, checker)
        self.arrelem2elem_class.append(prob_sol)


   #------------------------------------------------------------------------
    #
    def setupSumIfLT(self):

        prob = Problem("SumIfLT", 5, 1, Knob.max_probsz)  

        prob_gen = CntEQProbGenerator(prob)  # same probgen as count EQ
        checker = SumIfLTChecker(prob)

        prob_sol = ProbSolnPack(prob_gen, checker)
        self.arrelem2elem_class.append(prob_sol)

    def setupSumIfGT(self):

        prob = Problem("SumIfGT", 5, 1, Knob.max_probsz)  

        prob_gen = CntEQProbGenerator(prob)  # same probgen as count EQ
        checker = SumIfGTChecker(prob)

        prob_sol = ProbSolnPack(prob_gen, checker)
        self.arrelem2elem_class.append(prob_sol)


    #------------------------------------------------------------------------
    #
    def setupSumIfEQ(self):

        prob = Problem("SumIfEQ", 5, 1, Knob.max_probsz)  

        prob_gen = CntEQProbGenerator(prob)  # same probgen as count EQ
        checker = SumIfEQChecker(prob)

        prob_sol = ProbSolnPack(prob_gen, checker)
        self.arrelem2elem_class.append(prob_sol)


    #------------------------------------------------------------------------
    #
    def setupScaledSum(self):

        prob = Problem("ScaledSum", 5, 1, Knob.max_probsz) 

        prob_gen = CntEQProbGenerator(prob)  # same probgen as count EQ
        checker = ScaledSumChecker(prob)

        prob_sol = ProbSolnPack(prob_gen, checker)
        self.arrelem2elem_class.append(prob_sol)


    #------------------------------------------------------------------------
    # setup the 'Max' finding problem. All new problems must be setup the
    # same way
    #
    def setupMax(self):

        # first, create a new problem
        #
        prob = Problem("Max", 5, 1, Knob.max_probsz)    
        
        prob_gen = MaxProbGenerator(prob)   # create the problem generator ..
        checker = MaxChecker(prob)          # ... and checker

        # Now, package the problem generator and checker together and add
        # it to corresponding class of problems
        #
        prob_sol = ProbSolnPack(prob_gen, checker)
        self.arr2elem_class.append(prob_sol)


    #------------------------------------------------------------------------
    #
    def setupMaxIndex(self):

        # first, create a new problem
        #
        prob = Problem("MaxIndex", 5, 1, Knob.max_probsz)    
        
        prob_gen = MaxProbGenerator(prob)   # create the problem generator ..
        checker = MaxIndexChecker(prob)          # ... and checker

        # Now, package the problem generator and checker together and add
        # it to corresponding class of problems
        #
        prob_sol = ProbSolnPack(prob_gen, checker)
        self.arr2elem_class.append(prob_sol)

        if (True):

            inarr = [2, 23, 43, 12, 13, 14]
            prob.setInput(0, inarr)
            prob.setOutput(0, 2)

            suc = checker.check()
            if (Dbg.manual_check > 0):
                print("MaxInd Manual check out: ", suc)



    #------------------------------------------------------------------------
    #
    def setupLastIndexOf(self):

        # first, create a new problem
        #
        prob = Problem("LastIndOf", 5, 1, Knob.max_probsz)    
        
        prob_gen = IndexOfProbGenerator(prob)  # create the problem generator
        checker = LastIndexOfChecker(prob)          # ... and checker

        # Now, package the problem generator and checker together and add
        # it to corresponding class of problems
        #
        prob_sol = ProbSolnPack(prob_gen, checker)
        self.arrelem2elem_class.append(prob_sol)

        if (True):

            inarr = [2, 23, 43, 12, 13, 43, 14]
            prob.setInput(0, inarr)
            prob.setInput(1, 43)
            prob.setOutput(0, 5)

            suc = checker.check()
            if (Dbg.manual_check > 0):
                print("LastInd Manual check out: ", suc)


    #------------------------------------------------------------------------
    #
    def setupRemoveLast(self):

        # first, create a new problem
        #
        prob = Problem("RemoveL", 5, 1, Knob.max_probsz) 
        
        prob_gen = RemoveProbGenerator(prob)  # create the problem generator
        checker = RemoveLastChecker(prob)          # ... and checker

        # Now, package the problem generator and checker together and add
        # it to corresponding class of problems
        #
        prob_sol = ProbSolnPack(prob_gen, checker)
        self.arrelem2arr_class.append(prob_sol)

        if (True):

            inarr = [2, 23, 43, 12, 13, 43, 14]
            outarr = [2, 23, 43, 12, 13, 14]
            prob.setInput(0, inarr)
            prob.setInput(1, 43)
            prob.setOutput(0, outarr)

            suc = checker.check()
            if (Dbg.manual_check > 0):
                print("RemoveLast Manual check out: ", suc)

    #------------------------------------------------------------------------
    #
    def setupRemoveFirst(self):

        # first, create a new problem
        #
        prob = Problem("RemoveF", 5, 1, Knob.max_probsz) 
        
        prob_gen = RemoveProbGenerator(prob)  # create the problem generator
        checker = RemoveFirstChecker(prob)          # ... and checker

        # Now, package the problem generator and checker together and add
        # it to corresponding class of problems
        #
        prob_sol = ProbSolnPack(prob_gen, checker)
        self.arrelem2arr_class.append(prob_sol)

        if (True):

            inarr = [2, 23, 43, 12, 13, 43, 14]
            outarr = [2, 23, 12, 13, 43, 14]

            prob.setInput(0, inarr)
            prob.setInput(1, 43)
            prob.setOutput(0, outarr)

            suc = checker.check()
            if (Dbg.manual_check > 0):
                print("RemoveFirst Manual check out: ", suc)


    #------------------------------------------------------------------------
    #
    def setupFirstIndexOf(self):

        # first, create a new problem
        #
        prob = Problem("FirstIndOf", 5, 1, Knob.max_probsz) 
        
        prob_gen = IndexOfProbGenerator(prob)  # create the problem generator
        checker = FirstIndexOfChecker(prob)          # ... and checker

        # Now, package the problem generator and checker together and add
        # it to corresponding class of problems
        #
        prob_sol = ProbSolnPack(prob_gen, checker)
        self.arrelem2elem_class.append(prob_sol)

        if (True):

            inarr = [2, 23, 43, 12, 13, 43, 14]
            prob.setInput(0, inarr)
            prob.setInput(1, 43)
            prob.setOutput(0, 2)

            suc = checker.check()
            if (Dbg.manual_check > 0):
                print("FirstInd Manual check out: ", suc)


    #------------------------------------------------------------------------
    #
    def setupMin(self):

        prob = Problem("Min", 5, 1, Knob.max_probsz)    

        prob_gen = MaxProbGenerator(prob)   # use the same generator as Max
        checker = MinChecker(prob)          # use minimum checker

        prob_sol = ProbSolnPack(prob_gen, checker)
        self.arr2elem_class.append(prob_sol)


    #------------------------------------------------------------------------
    #
    def setupSum(self):

        prob = Problem("Sum", 5, 1, Knob.max_probsz)    

        prob_gen = MaxProbGenerator(prob)   # use the same generator as Max
        checker = ArrSumChecker(prob)          # use sum checker

        prob_sol = ProbSolnPack(prob_gen, checker)
        self.arr2elem_class.append(prob_sol)


    #------------------------------------------------------------------------
    #
    def setupSumOfSquares(self):

        prob = Problem("SumOfSq", 5, 1, Knob.max_probsz)  

        prob_gen = MaxProbGenerator(prob)   # use the same generator as Max
        checker = SumOfSquaresChecker(prob)          # use sum checker

        prob_sol = ProbSolnPack(prob_gen, checker)
        self.arr2elem_class.append(prob_sol)


    #------------------------------------------------------------------------
    #
    def setupAvg(self):

        # create 'avg' problem
        #
        prob = Problem("Avg", 5, 1, 2*Knob.max_probsz) 

        prob_gen = ArrAvgProbGenerator(prob)   # use the same generator as Max
        checker = AvgChecker(prob)          # use avg checker

        prob_sol = ProbSolnPack(prob_gen, checker)
        self.arr2elem_class.append(prob_sol)


        if (True):

            inarr = [10, 20, 30, 40, 50, 60, 72]
            prob.setInput(0, inarr)
            prob.setOutput(0, 282//7)

            suc = checker.check()
            print("Avg check out: ", suc)



    #------------------------------------------------------------------------
    #
    def setupScaledAvg(self):

        prob = Problem("ScaledAvg", 5, 1, 2*Knob.max_probsz) 

        prob_gen = CntEQProbGenerator(prob)   
        checker = ScaledAvgChecker(prob)         

        prob_sol = ProbSolnPack(prob_gen, checker)
        self.arrelem2elem_class.append(prob_sol)




    #------------------------------------------------------------------------
    #
    def setupArrLen(self):

        prob = Problem("Len", 5, 1, Knob.max_probsz)    

        prob_gen = MaxProbGenerator(prob)    # use the same generator as Max
        checker = ArrLenChecker(prob)        # use arr len checker

        prob_sol = ProbSolnPack(prob_gen, checker)
        self.arr2elem_class.append(prob_sol)


    #------------------------------------------------------------------------
    #
    def setupSortDescend(self):

        prob = Problem("SortDesc", 5, 1, Knob.max_probsz//2)  

        prob_gen = SortProbGenerator(prob)  
        checker = SortCheckerDescend(prob)       

        prob_sol = ProbSolnPack(prob_gen, checker)
        self.arr2arr_class.append(prob_sol)

        # manual test of Checker to verify the Checker is correct
        #
        if (False):

            inarr = [5, 9, 8, 3, 2]
            outarr = [9, 8, 5, 3, 2]

            prob.setInput(0, inarr)
            prob.setOutput(0, outarr)

            suc = checker.check()
            if (Dbg.manual_check > 0):
                print("Manual check out: ", suc)

    #------------------------------------------------------------------------
    #
    def setupSortAscend(self):

        prob = Problem("SortAsc", 5, 1, Knob.max_probsz//2)  

        prob_gen = SortProbGenerator(prob)  
        checker = SortCheckerAscend(prob) 

        prob_sol = ProbSolnPack(prob_gen, checker)
        self.arr2arr_class.append(prob_sol)

    #------------------------------------------------------------------------
    #
    def setupReverse(self):

        prob = Problem("ReverseArr", 5, 1, Knob.max_probsz)  

        prob_gen = SortProbGenerator(prob)        # same as for sort
        checker = reverseChecker(prob) 

        prob_sol = ProbSolnPack(prob_gen, checker)
        self.arr2arr_class.append(prob_sol)

    #------------------------------------------------------------------------
    #
    def setupArrAdd(self):

        prob = Problem("AddArrays", 5, 1, Knob.max_probsz)  

        prob_gen = ArrAddProbGenerator(prob)      
        checker = ArrAddChecker(prob) 

        prob_sol = ProbSolnPack(prob_gen, checker)
        self.twoarr2arr_class.append(prob_sol)

        # manual test of Checker to verify the Checker is correct
        #
        if (False):

            inarr1 = [5,  9, 8, 3, 2]
            inarr2 = [3,  9, 1, 5, 0]
            outarr = [8, 18, 9, 8, 2]

            prob.setInput(0, inarr1)
            prob.setInput(1, inarr2)
            prob.setOutput(0, outarr)

            suc = checker.check()
            if (Dbg.manual_check > 0):
                print("Manual check out: ", suc)




    #------------------------------------------------------------------------
    #
    def setupArrMult(self):

        prob = Problem("MultArrays", 5, 1, Knob.max_probsz)  

        prob_gen = ArrAddProbGenerator(prob)
        checker = ArrMultChecker(prob)

        prob_sol = ProbSolnPack(prob_gen, checker)
        self.twoarr2arr_class.append(prob_sol)



    #------------------------------------------------------------------------
    #
    def setupDotProduct(self):

        prob = Problem("DotProd", 5, 1, Knob.max_probsz)  

        prob_gen = DotProdProbGenerator(prob)      
        checker = DotProductChecker(prob) 

        prob_sol = ProbSolnPack(prob_gen, checker)
        self.twoarr2elem_class.append(prob_sol)

        if (True):

            vec1 = [5, 9, 8, 3, 2]
            vec2 = [1, 1, 1, 1, 1]
            out =  27

            prob.setInput(0, vec1)
            prob.setInput(1, vec2)
            prob.setOutput(0, out)

            suc = checker.check()
            if (Dbg.manual_check > 0):
                print("DotProd Manual check out: ", suc)


    #------------------------------------------------------------------------
    #
    def setupSumAdd(self):

        prob = Problem("SumAdd", 5, 1, Knob.max_probsz)  

        prob_gen = DotProdProbGenerator(prob)      
        checker = SumAddChecker(prob) 

        prob_sol = ProbSolnPack(prob_gen, checker)
        self.twoarr2elem_class.append(prob_sol)


    #------------------------------------------------------------------------
    #
    def setupMatVec(self):

        prob = Problem("MatVecMult", 5, 1, 20)  

        prob_gen = MatVecProbGenerator(prob)      
        checker = MatVecChecker(prob) 

        prob_sol = ProbSolnPack(prob_gen, checker)
        self.aOa_arr2arr_class.append(prob_sol)

        # manual test of Checker to verify the Checker is correct
        #
        if (True):

            matrix = [[5, 9, 8, 3, 2], [1,2,3,4,5]]
            vec = [1, 1, 1, 1, 1]
            outarr = [27, 15]

            prob.setInput(0, matrix)
            prob.setInput(1, vec)
            prob.setOutput(0, outarr)

            suc = checker.check()
            if (Dbg.manual_check > 0):
                print("Mat Vec Manual check out: ", suc)


    #------------------------------------------------------------------------
    #
    def setupMatAddVec(self):

        prob = Problem("MatVecAdd", 5, 1, 20)  

        prob_gen = MatVecProbGenerator(prob)      
        checker = MatAddVecChecker(prob) 

        prob_sol = ProbSolnPack(prob_gen, checker)
        self.aOa_arr2arr_class.append(prob_sol)

        # manual test of Checker to verify the Checker is correct
        #
        if (True):

            print("Testing Mat Add Vec:")

            matrix = [[5, 9, 8, 3, 2], [1,2,3,4,5]]
            vec = [1, 1, 1, 1, 1]
            outarr = [32, 20]

            prob.setInput(0, matrix)
            prob.setInput(1, vec)
            prob.setOutput(0, outarr)

            suc = checker.check()
            if (Dbg.manual_check > 0):
                print("Mat Add Vec Manual check out: ", suc)


    #------------------------------------------------------------------------
    #
    def setupAddElem(self):

        prob = Problem("AddToArr", 5, 1, Knob.max_probsz)  

        prob_gen = AddElemProbGenerator(prob)      
        checker = AddElemChecker(prob) 

        prob_sol = ProbSolnPack(prob_gen, checker)
        self.arrelem2arr_class.append(prob_sol)

    #------------------------------------------------------------------------
    #
    def setupSubElem(self):

        prob = Problem("SubFromArr", 5, 1, Knob.max_probsz)  

        prob_gen = AddElemProbGenerator(prob)      
        checker = SubElemChecker(prob) 

        prob_sol = ProbSolnPack(prob_gen, checker)
        self.arrelem2arr_class.append(prob_sol)


    #------------------------------------------------------------------------
    #
    def setupScaleArr(self):

        prob = Problem("ScaleArr", 5, 1, Knob.max_probsz)  

        prob_gen = AddElemProbGenerator(prob)  # same probgen as adding elem
        checker = ScaleArrChecker(prob) 

        prob_sol = ProbSolnPack(prob_gen, checker)
        self.arrelem2arr_class.append(prob_sol)


    #------------------------------------------------------------------------
    #
    def setupIsInArr(self):

        prob = Problem("IsInArr", 40, 1, 2*Knob.max_probsz)  

        prob_gen = IsInArrProbGenerator(prob)  # same probgen as adding elem
        checker = IsInArrChecker(prob) 

        prob_sol = ProbSolnPack(prob_gen, checker)
        self.arrelem2bool_class.append(prob_sol)



    #------------------------------------------------------------------------
    # Routine to analyze solutions for their complexity and pick one
    #
    def analyzeSolns(self, pname2solnslist):
        

        pname_list = pname2solnslist.keys()       # list of all keys (pnames)

        # create a dictionary to record the least complex solution for
        # each problem
        #
        pname2mincost_psol = dict()

        # Go through each problem 
        #
        for pname, psol_list in pname2solnslist.items():

            if (len(psol_list) == 0):
                print("WARN: No solutions for problem ", pname)
                continue

            min_cost_metric = SolnMetrics(psol_list[0])
            min_len_metric = SolnMetrics(psol_list[0])

            callee_cnts = dict()        # callee counts
            parent_cnts = dict()        # parent counts

            # Find least complex / sorted solution out of all solutions
            # found for this problem
            #
            for psol in psol_list:      # go through each problems/soln

                psol_metric = SolnMetrics(psol)

                if (psol_metric.complexity < min_cost_metric.complexity):
                    min_cost_metric = psol_metric

                if (psol_metric.len < min_len_metric.len):
                    min_len_metric = psol_metric
                

                # record/increment callee counts of functions called 
                # by this psol
                #
                for callee in psol_metric.callees:
                    if (callee_cnts.get(callee) != None):
                        callee_cnts[callee] += 1
                    else:
                        callee_cnts[callee] = 1

                # record/increment parent counts
                #
                parent_name = psol.sol_stats.parent_name
                if (parent_cnts.get(parent_name) != None):
                    parent_cnts[parent_name] += 1
                else:
                    parent_cnts[parent_name] = 1


            # Print least complex soln and STAT rec
            #
            print("Least Complex Soln for ", pname, "====================")
            print(min_cost_metric.psol.sol_func_call_expr.func_def_str)
            min_cost_metric.psol.sol_stats.dump()
            #
            # record least complex soln for pname
            #
            pname2mincost_psol[ pname ] = min_cost_metric.psol
            

            # Print least len soln and STAT rec
            #
            print("Least Length Soln for ", pname, ":")
            print(min_len_metric.psol.sol_func_call_expr.func_def_str)
            min_len_metric.psol.sol_stats.dump()


            # Print callee stats -- ie., how many times this function calls
            # each of the other functions in the class. This is printed as
            # two lines in comma separated format
            #
            print("\nCALLEES:,\tForProb", end='') 
            for p in pname_list:          
                print(",\t", p, end='')
            print(",\tTotSolns")

            print("CALLEE_CNTS:,\t", pname, end='')    
            for p in pname_list:          
                cnt = callee_cnts.get(p)
                if (cnt == None):
                    cnt = 0
                print(",\t", cnt, end='')
            print(",\t", len(psol_list), "\n")


            # Print parent stats -- i.e., parents of soln. This is printed as
            # two lines in comma separated format
            #
            print("\nPARENTS:,\tForProb", end='') 
            for p in pname_list:          
                print(",\t", p, end='')
            print(",\tTotSolns")

            print("PARENT_CNTS:,\t", pname, end='')    
            for p in pname_list:          
                cnt = parent_cnts.get(p)
                if (cnt == None):
                    cnt = 0
                print(",\t", cnt, end='')
            print(",\t", len(psol_list), "\n")
        
        # STEP: .......................................................
        # At this point we have found least complex solutions for all problems
        # Now compose entire solution (all funcs) using least complex solutions
        #
        #
        if (Knob.compose_least_complex):
            self.printLeastComplexSolns(pname2mincost_psol)

    #------------------------------------------------------------------------
    # Routine to print each solution using least complex solutions found
    # for each function
    #
    def printLeastComplexSolns(self, pname2mincost_psol):

        print("============= Composed Least Complx Solns ================\n")

        # Go through each problem
        #
        for pname, psol in pname2mincost_psol.items():
            
            self.solns_printed = dict()           # init dict 

            print("----- Full Least Complex Solution for ", pname, " -----\n")
            #
            self.printLeastComplexFunc(pname, pname2mincost_psol)
            

    #------------------------------------------------------------------------
    # Recursive routine to print each cleaned function in a function
    # call tree. 
    # ASSUMPTION: This assumes that there are no recursive functions
    #
    def printLeastComplexFunc(self, pname, pname2mincost_psol):
        
        psol = pname2mincost_psol.get(pname)

        # If there is no solution in the dict (this should not happen)
        #
        if (psol == None):
            print("ERROR: psol not found for:", pname)
            return

        # If we already printed, don't print again
        # This happens if least complex solution for this problem
        # uses another least complex one, which causes recursion
        # In such a case, we need to discard this and look for the
        # next one. Currently, we just look for one manually.
        #
        if (self.solns_printed.get( pname ) == True):
            print("WARN: Recursion! Discard this. Find another manually");
            return
        else:
            self.solns_printed[ pname ] = True

        func = psol.sol_clean_func  # function 


        # First, go through each estmt in function and
        # print any callees
        #
        for estmt in func.estmts:

            if estmt.isFuncCall():                # if a func call

                fname = estmt.func_name           # name of the func
                
                self.printLeastComplexFunc(fname, pname2mincost_psol)

        # Now, we have printed all callees. So, print this function
        # We need to print only the last function in func_def_str
        #
        func_str = psol.sol_func_call_expr.func_def_str
        last_ind = func_str.rfind("def " + pname)     # find last function def
        last_func_str = func_str[last_ind:]   # last function
        #
        # print the last occurring function
        #
        print("\n", last_func_str)
       





##############################################################################
# Class to evaluate 'complexity' metrics for a given solution
##############################################################################
class SolnMetrics:

    func_weight = 50                    # cost of a func call
    for_idiom_weight = 20               # cost of for block
    if_idiom_weight = 10                # cost of if block
    stmt_weight = 2                     # cost of any other stmt

    
    def __init__(self, psol):

        self.len = -1                   # length of func (stmt count)
        self.complexity = -1            # complexity
        self.psol = psol                # psol
        self.callees = set()            # funcs this sol calls

        # calculate the complexity of this psol
        #
        self.calcComplexity(psol)

    # Method for calculating the complexity of a solution
    #
    def calcComplexity(self, psol):

        func = psol.sol_clean_func

        self.len = len(func.estmts)               # length of stmts

        cost = 0

        for estmt in func.estmts:

            if estmt.isFuncCall():                # if a func call
                cost += SolnMetrics.func_weight
                self.callees.add(estmt.func_name) # add to the set

            elif estmt.isBlkHead():               # if this is an idiom
                if (estmt.isIfExpr()):
                    cost += SolnMetrics.if_idiom_weight
                elif (estmt.isForExpr()):
                    cost += SolnMetrics.for_idiom_weight
            else:
                cost += SolnMetrics.stmt_weight

        self.complexity = cost


#========================== Per Process Starting Point ========================
#
def procMain(procid, outq, all_psol_classes, epoch):

    # First, redirect stdout to a file
    #
    mode = "w" if (epoch==0) else "a"

    stdout_root = getLogDir() + "/" + Knob.log_fname_root

    sys.stdout = open(stdout_root + "." + str(procid) + ".txt", mode)

    print("eeeeeeeeeee PROCESS EPOCH: ", epoch, "eeeeeeeeeeeeee")
    print("Start Time:", getTimeStamp(True))

    # Create a new framework for this epoch and set all_psol_classes
    #
    framework = Framework(procid, epoch)
    #
    framework.all_psol_class_list = all_psol_classes

    # Evolve it
    # Change stmt limit based on the epoch. Start with Knob.stmt_limit
    #       and increase 
    #
    stmt_limit = Knob.stmt_limit_start + (epoch // Knob.epochs_per_stmt_incr)
    #
    framework.evolve(stmt_limit)

    # At the end of evolution, add all problems/solutions to outq
    #
    outq.put((procid, framework.all_psol_class_list))   # add to output queue

    GlobStat.dump(procid)               # dump global stats


# ========================= Multi-Processor Support =========================


#############################################################################
# This class represents a checkpoint of all solutions for all problems we
# have found. 
# Strategy: We take 'delta' checkpoints, either at the end of a run or 
#   at regular intervals (of epochs). All nodes dump their delta checkpoints.
#   When restoring a checkpoint, we need to read all delta checkpoints, taken
#   from the beginning, which are present in Knob.chkpt_dir.
# 
# Delta Checkpoint Filename Format: 
#    <nodename>.<epoch>.<randomid>.pkl
#
#############################################################################
class Chkpt:

    # enumerations for file handling
    #
    FileExt = "pkl"

    # File format regular expression
    #
    FileFormatRegEx = r'(\w+)\.(\d+)\.(\d+)\.' + FileExt


    #------------------------------------------------------------------------
    # Creates a new checkpoint, which is a mapping (dict) between a problem
    # name and its solution list
    #
    def newChkpt(orig_psol_class_list):

        pname2solnslist = dict()

        # Now, create an empty checkpoint for each problem in each class
        #
        for prob_class_list in orig_psol_class_list: 
            for psol in prob_class_list:
                pname2solnslist[ psol.getName() ] = list()


        return ( pname2solnslist )


    #------------------------------------------------------------------------
    def getChkptFName(nodename, epoch):

        randid = random.randint(1000000, 1000000000)        # random id

        outfname = nodename + "." + str(epoch) + "." + str(randid) + "." + \
            Chkpt.FileExt

        #assert that outfname confirms to file name std we use
        #
        assert( re.search(Chkpt.FileFormatRegEx, outfname) )

        return Knob.chkpt_dir + "/" + outfname


    #------------------------------------------------------------------------
    # save a checkpoint of all solutions we have found so far, passed in with
    # pname2solnslist, which is a map (dict) from prob names to solution lists
    #
    def saveChkpt(nodename, epoch, pname2solnslist):

        # Note: we write to a tmp file first, and then rename it to the
        #       required file name (outfname). We do this to prevent some
        #       other node from reading a file that is not completely written.
        #
        outfname_full = Chkpt.getChkptFName(nodename, epoch)
        tmpfname = outfname_full + ".part"

        with open(tmpfname, 'wb') as output:

            # dump with highest std
            #
            pickle.dump(pname2solnslist, output, -1)  
            
            os.rename(tmpfname, outfname_full)
            Dbg.prn(Dbg.chkpt, "Saved checkpoint " + outfname_full)

    #----------------------------------------------------------------------
    # Read a checkpoint, from multiple checkpoint files (from each node)
    # into the map (dict) pname2solnslist
    # PRE: pname2solnslist must have all the problem *names* but the 
    # solution lists must be *empty*
    #
    def readChkpt(pname2solnslist):

        # read all the 'delta' checkpoint files
        #
        flist = glob.glob(Knob.chkpt_dir + "/*." + Chkpt.FileExt)

        if (len(flist) == 0):
            Dbg.warn("No checkpoint files found for reading")

        for fname in flist:

            with open(fname, 'rb') as input:

                delta_chkpt = pickle.load(input)

                Dbg.prn(Dbg.chkpt, "Merging chkpt from " +  fname)

                Chkpt.mergeSolnsFromChkpt(delta_chkpt, pname2solnslist)

 
    #----------------------------------------------------------------------
    # Merge solutions in checkpoint dictionary 'delta_chkpt' into the dict
    # we are constructing 'pname2solnslist'
    #
    def mergeSolnsFromChkpt(delta_chkpt, pname2solnslist):

        # Go through each problem in pname2solnslist and insert to its list
        #
        for pname, sol_list in pname2solnslist.items():
        
            # if the problem exists in the checkpoint as well
            # append those solutions to sol_list
            #
            if pname in delta_chkpt:

                chkpt_list = delta_chkpt[pname]

                for chkpt_soln in chkpt_list:     # for each soln in chkpt list

                    if len(sol_list) < Knob.max_solns4prob:
                        sol_list.append(chkpt_soln)

                if (Dbg.chkpt > 0):               # debug print
                    t = len(sol_list)
                    s = len(chkpt_list)
                    print("\t-found ", s, " solns for ", pname, ". Tot:", t)


                    

#***********************  Startup/Main Routines ****************************


#------------------------------------------------------------------------
# Helper func: Accept solns based on first come first served basis
#------------------------------------------------------------------------
def acceptNewSoln(newpsol, exist_list):


    if (len(exist_list) < Knob.max_solns4prob):

        # if there is space left in the list
        #
        exist_list.append(newpsol)            # append to list
        #print("Appended new soln to list")
        return True
    else:
        return False


#----------------------------------------------------------------------
# Main steps
#----------------------------------------------------------------------
def main():

    # First create a ProbSolnList, containing all problems, and empty solns
    #
    orig_all_prob_soln_obj = AllProbSolnPack()
    #
    orig_psol_class_list = orig_all_prob_soln_obj.all_psol_classes

    # Now create processes to launch
    #
    #num_procs = 1
    num_procs = mp.cpu_count() // Knob.hyperthreads_per_cpu
    nodename = socket.gethostname()
    print("Node:", nodename, " Using ", num_procs, " CPUs")

    # This dictionary is a mapping between a problem name and a list of 
    # all solutions (psols) found for that problem, on different ranks. 
    # It is created by Chkpt class because it is data structure used for
    # checkpointing as well
    #
    pname2solnslist = Chkpt.newChkpt(orig_psol_class_list)

    # Create a 'delta' list of solutions. This is used for taking only
    # 'delta checkpoints' -- i.e., solutions found from previous chkpt
    #
    pname2soln_delta = Chkpt.newChkpt(orig_psol_class_list)

    # if we are starting from a checkpoint, read it. This can be from a
    # single node checkpoint or a cluster checkpoint
    #
    if (Knob.start_from_chkpt):
        Chkpt.readChkpt(pname2solnslist)
    

    for epoch in range(Knob.num_epochs):

        print("\neeeeeeeeeeeeee Parallel EPOCH: ", epoch, "eeeeeeeeeeeeeeee\n")

        
        # if dumping of checkpoints is requested, dump it after every N
        # epochs, where N is Knob.epochs4chkpt. After dumping, if we are
        # running on a cluster, we can read checkpoints taken by other nodes
        # as well. 
        #
        if (Knob.dump_chkpts):
            
            if ((epoch > 0) and ((epoch % Knob.epochs4chkpt) == 0)):

                # Note. We dump the 'delta' checkpoint. Then we crate a
                # new one
                #
                Chkpt.saveChkpt(nodename, epoch, pname2soln_delta)
                pname2soln_delta = Chkpt.newChkpt(orig_psol_class_list)

                # We read ALL delta checkpoints into pname2solnslist
                #
                if (Knob.read_cluser_chkpt):
                    pname2solnslist = Chkpt.newChkpt(orig_psol_class_list)
                    Chkpt.readChkpt(pname2solnslist)


        # Create a queue for communication. 
        # Each process will output to this queue
        #
        outq = mp.Queue()             # creates a queue for communication

        # Now create all processes, with target function and args
        #
        procs = list()                # create processes
        #
        for p in range(num_procs):

            # STEP: First, create a all psol class list to send to proc p
            #       If there are already solved solutions for a problem, pick
            #       one at random
            #
            cur_psol_class_list = list()    # all psol class list for p
            #
            for orig_psols_in_class in orig_psol_class_list: 

                # Create a new list for this class
                #
                cur_psols_in_class = list()        
                cur_psol_class_list.append( cur_psols_in_class )  

                # Now, go through all problems, and find existing solutions
                #
                for orig_psol in orig_psols_in_class:
                
                    pname = orig_psol.getName()
                    solns_found_for_prob = pname2solnslist[pname]
                    num_existing_solns = len(solns_found_for_prob)

                    # if there are existing solutions in the dictionary, 
                    # pick one at random. If there are none, just use the
                    # original problem 
                    #
                    # We must find a new soln if there are no solns. We elect
                    # to find a new soln, if we have not reached the soln 
                    # limit AND a coin toss succeeds
                    #
                    find_new = (num_existing_solns == 0) or \
                    ((num_existing_solns < Knob.max_solns4prob) and \
                     (random.randint(1,100) <= Knob.find_more_solns_odds))   
                    
                    if (find_new):
                        cur_psols_in_class.append(orig_psol)
                    else:
                        ind = random.randint(0, num_existing_solns-1) 
                        cur_psols_in_class.append(solns_found_for_prob[ind])

                assert(len(cur_psols_in_class) == len(orig_psols_in_class))

            assert(len(cur_psol_class_list) == len(orig_psol_class_list))

            # Now, we have all the psol classes and their problems to be 
            # sent to this processor. Set args this process and create it
            #
            pargs = (p, outq, cur_psol_class_list, epoch)
            procs.append( mp.Process(target=procMain, args=pargs) )

        # STEP:
        # Now, we have created all processes. Start them one by one
        #
        for proc in procs:            # fork all the processes in procs
            proc.start()


        # read results from queue and append to lists
        #
        retobjs = list()
        retpids = list()

        for p in procs:
            (pid, retobj) = outq.get()
            retobjs.append(retobj)
            retpids.append(pid)
            lastprocid = pid

        # Note: we have to join after reading the results from the queue
        #
        for proc in procs:            # Wait for all processes to join
            proc.join()


        # Whether we have solved all problems
        #
        all_probs_solved = True

        # Go through each all_prob_soln objects returned by each process
        #
        for msg_num in range(len(retobjs)):

            ret_psol_classes = retobjs[msg_num]
            ret_pid = retpids[msg_num]

            if (Dbg.multi_proc > 0):
                print("\nEpoch:", epoch, " Rank:", ret_pid, " .........")

            # Var to detect whether all problems are solved
            #
            all_probs_solved = True             

            # Go through each class
            #
            for class_id in range(len(ret_psol_classes)):

                # returned and original (existing) class lists
                #
                ret_psol_list = ret_psol_classes[class_id]   
                orig_psol_list = orig_psol_class_list[class_id]

                # Go through each problem/solution in the class
                #
                for psol_id in range(len(ret_psol_list)):

                    ret_psol = ret_psol_list[psol_id]       # returned psol
                    orig_psol = orig_psol_list[psol_id]     # original psol
                    pname = ret_psol.getName()              # name of prob

                    if (Dbg.multi_proc > 0):
                        print("Solns for ", pname, end=': ')

                    # get existing list of solns for this problem
                    #
                    exist_list = pname2solnslist[pname]
                    num_solns = len(exist_list) 
                    new_soln_accepted = False

                    # If the returned psol has a newly found solution, add
                    # it to pname2solnslist, where all solutions found for
                    # a problem is recorded
                    #
                    if (ret_psol.is_soln_new):

                        assert(ret_psol.sol_func != None)

                        Dbg.prnwb(Dbg.multi_proc, "\tNew")

                        # See whether we accept this new solution into the
                        # exist_list, based on size of the exist_list
                        #
                        if acceptNewSoln(ret_psol, exist_list):

                            new_soln_accepted = True
                            num_solns += 1
                            ret_psol.is_soln_new = False

                            # Also, append to 'delta' dict for delta checkpt
                            #
                            pname2soln_delta[pname].append(ret_psol)

                        else:
                            pass

                        Dbg.prn(Dbg.multi_proc, "\tTot:" + str(num_solns))

                        # print new solution
                        #
                        if (num_solns == 1):
                            print(ret_psol.sol_func_call_expr.func_def_str)
                            ret_psol.sol_stats.dump()
                    else:
                        Dbg.prn(Dbg.multi_proc, 
                                "\tNoNew\tTot:" + str(num_solns))

                    # if there are no solutions found yet
                    #
                    if (len(exist_list) == 0):
                        all_probs_solved = False

            # flush after every complete message
            #
            # sys.stdout.flush()

        # if all problems solved end the simulation
        #
        if (all_probs_solved):
            print("!!!SUCCESS - All Problems in All Classes Solved!!!")

            # End run if requested to end before epoch count end
            #
            if Knob.end_when_solutions_found:
                break
            else:
                print("Continuing until epoch count end")
            


    # if dumping of checkpoint is requested, do so here before we exit
    #
    if (Knob.dump_chkpts):
        Chkpt.saveChkpt(nodename, epoch, pname2soln_delta)

    print("\nEnd time:", getTimeStamp(True))

    # Analyze solutions and pick ones with least complexity
    # Redirect output from this point to a report file
    #
    if (Knob.print_least_complex_solns):
        sys.stdout.flush()
        sys.stdout = open(Knob.report_name, "w")
        orig_all_prob_soln_obj.analyzeSolns(pname2solnslist)



#----------------------------------------------------------------------------
# Helper method to get the log directory for a node
#
def getLogDir():
    nodename = socket.gethostname()
    mylogdir = Knob.log_dir_root + "." + nodename
    return mylogdir

#----------------------------------------------------------------------------
# Helper method to print time stamp
#
import time
import datetime
#
def getTimeStamp(print_tod=False):

    curtime = str(time.clock_gettime(time.CLOCK_MONOTONIC))
    if (print_tod):
        curtime += " " + str(datetime.datetime.now())        # if date/time

    return str(curtime)

#----------------------------------------------------------------------------
# Helper method to config for reporting (from checkpoints)
#
def setReportOptions():
    Knob.start_from_chkpt = True        # start with checkpoint
    Knob.num_epochs = 0                 # Do not evolve
    Knob.dump_chkpts = False            # no dumping of checkpoint
    Knob.report_name = Knob.full_report_name
    Knob.compose_least_complex = True   # compose least complex solutions
    Dbg.manual_check = 0                # no initial checking of checkers
    print("INFO: Reading all checkpoints from ", Knob.chkpt_dir) 


#======================== Main Entry Point ===================================

# Check for version (tested with only this version)
#
if (sys.version_info[0] < 3) or (sys.version_info[1] < 6):
    print("ERROR: Please run this script with Python version 3.6 or later")
    exit(1)


# Print start time
#
print("\nStart time:", getTimeStamp(True))

# Process command line args
#
if (len(sys.argv) > 1):                 # Read Group
        Knob.group2solve = int(sys.argv[1])  
        print("INFO: Setting Problem Group to ", Knob.group2solve)

if (len(sys.argv) > 2):                 # If reporting is requested
    cmd_opt = int(sys.argv[2])          # get cmd line option
    print("INFO: Entering reporting mode (from checkpoints)")
    setReportOptions()                  # set options for reporting

else:                                   # if not reporting mode
    # Create a unique log dir for each node
    #
    mylogdir = getLogDir()

    # Make directory for log files from each process
    #
    print("INFO: Creating ", mylogdir, " and ", Knob.chkpt_dir)
    os.system("mkdir -p " + mylogdir)


    # Note. We create only 1 shared checkpt directory for all nodes
    #
    os.system("mkdir -p " + Knob.chkpt_dir)


# call main
#
main()



