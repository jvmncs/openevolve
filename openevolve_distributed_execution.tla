---- MODULE OpenEvolve ----
EXTENDS Naturals, Sequences, TLC

(***************************************************************************)
(* CONSTANTS                                                              *)
(***************************************************************************)

CONSTANTS
  PopSize   \* required number of children per generation
           ,MaxGen    \* maximal number of generations we explore
           ,WorkerIds \* finite set of potential Modal workers
          
ASSUME PopSize >= 1
ASSUME MaxGen   >= 1
ASSUME WorkerIds /= {}

(***************************************************************************)
(* TYPES                                                                  *)
(***************************************************************************)

Program     == [ id        : Nat
                ,parentId  : Nat \cup {Nil}
                ,gen       : Nat
                ,score     : Real
                ]

\* Simple generator for unique program identifiers
ProgIdSeq(n) == n          \* (identity; TLC handles uniqueness because n is monotone)

(***************************************************************************)
(* STATE VARIABLES                                                        *)
(***************************************************************************)

VARIABLES
  genIdx          \* generation currently being produced
 ,popSize         \* population size for current generation (== PopSize)
 ,tasksScheduled  \* how many workers have been handed a parent
 ,tasksCommitted  \* how many children are already committed
 ,staging         \* set of Program produced this generation but not yet merged
 ,db              \* authoritative Program database (set of Program)
 ,running         \* TRUE when a generation is in progress
 ,pending         \* number of worker tasks that have been spawned and not yet finished
 ,sandboxState    \* dummy variable to show isolation (never leaves worker)

(***************************************************************************)
(* Helper predicates                                                      *)
(***************************************************************************)

GenerationComplete == running /\ (tasksCommitted = popSize)

CanRequestParent   == running /\ tasksScheduled < popSize

CanCommitChild     == running /\ tasksCommitted < popSize

NoGenInProgress    == ~running

(***************************************************************************)
(* INITIALISATION                                                         *)
(***************************************************************************)

Init ==
  /\ genIdx = 0
  /\ popSize = 0
  /\ tasksScheduled = 0
  /\ tasksCommitted = 0
  /\ staging = {}
  /\ db = {}           \* database initially empty
  /\ running = FALSE
  /\ pending = 0
  /\ sandboxState = [ w \in WorkerIds |-> Nil ]

(***************************************************************************)
(* ACTIONS                                                                *)
(***************************************************************************)

(**************** Controller behaviour ****************)

StartGeneration ==
  /\ NoGenInProgress
  /\ genIdx < MaxGen
  /\ popSize'         = PopSize
  /\ running'         = TRUE
  /\ tasksScheduled'  = 0
  /\ tasksCommitted'  = 0
  /\ staging'         = {}
  /\ UNCHANGED << db, pending, sandboxState, genIdx >>

BarrierCommitAndAdvance ==
  /\ GenerationComplete
  /\ db'         = db \cup staging           \* Hub writes database atomically
  /\ genIdx'     = genIdx + 1
  /\ running'    = FALSE
  /\ popSize'    = 0
  /\ tasksScheduled' = 0
  /\ tasksCommitted' = 0
  /\ staging'   = {}
  /\ UNCHANGED << pending, sandboxState >>

(**************** Worker-side behaviour ****************)

RequestParent(w) ==
  /\ w \in WorkerIds
  /\ CanRequestParent
  /\ tasksScheduled' = tasksScheduled + 1
  /\ pending'        = pending + 1
  /\ UNCHANGED << genIdx, popSize, tasksCommitted, running,
                  db, staging, sandboxState >>
                  
\* The LLm generation + evaluation are abstracted to creation of a fresh Program.
DoWorkAndCommit(w) ==
  /\ w \in WorkerIds
  /\ pending > 0
  /\ CanCommitChild
  /\ pending'        = pending - 1
  /\ tasksCommitted' = tasksCommitted + 1
  
  /\ LET newProg ==
        [ id       |-> ProgIdSeq(Cardinality(db) + Cardinality(staging) + 1)
        , parentId |-> 0                      \* abstract parent
        , gen      |-> genIdx
        , score    |-> RandomElement(0..100)  \* nondeterministic score
        ]
     IN staging'   = staging \cup { newProg }
     
  /\ sandboxState' = [ sandboxState EXCEPT ![w] = "discarded" ]
  /\ UNCHANGED << genIdx, popSize, tasksScheduled, running, db >>

\* Work can also return early if generation became full before commit.
WorkerExitEarly(w) ==
  /\ w \in WorkerIds
  /\ pending > 0
  /\ ~running
  /\ pending' = pending - 1
  /\ UNCHANGED << genIdx, popSize, tasksScheduled, tasksCommitted, running,
                  db, staging, sandboxState >>

(***************************************************************************)
(* NEXT-STATE RELATION                                                    *)
(***************************************************************************)

Next ==
  \/ StartGeneration
  \/ BarrierCommitAndAdvance
  \/ \E w \in WorkerIds : RequestParent(w)
  \/ \E w \in WorkerIds : DoWorkAndCommit(w)
  \/ \E w \in WorkerIds : WorkerExitEarly(w)

(***************************************************************************)
(* INVARIANTS                                                             *)
(***************************************************************************)

Inv_TaskBounds ==
  /\ tasksScheduled <= popSize
  /\ tasksCommitted <= tasksScheduled

Inv_BarrierCorrectness ==
  running \/ (tasksCommitted = 0 /\ tasksScheduled = 0)

Inv_SingleWriter ==
  \A w \in WorkerIds :
      (sandboxState[w] # sandboxState'[w]) => db' = db
  \* Workers never modify db; only BarrierCommitAndAdvance does.

Inv_SandboxIsolation ==
  \A w \in WorkerIds :
      sandboxState[w] = Nil => sandboxState'[w] = Nil
  \* Isolation: outside of the worker step the sandbox variable for that worker
    must not spontaneously change.

(***************************************************************************)
(* SPECIFICATION                                                          *)
(***************************************************************************)

Spec == Init /\ [][Next]_<< genIdx, popSize, tasksScheduled, tasksCommitted,
                   staging, db, running, pending, sandboxState >>

THEOREM TypeOK ==
  Spec => []Inv_TaskBounds

THEOREM BarrierSafe ==
  Spec => []Inv_BarrierCorrectness

THEOREM SingleWriterSafe ==
  Spec => []Inv_SingleWriter

THEOREM SandboxSafe ==
  Spec => []Inv_SandboxIsolation

(***************************************************************************)
(* TLC CONFIGURATION HINTS                                                *)
(***************************************************************************)
\* Suggested default values when running with TLC:
\*   PopSize   = 3
\*   MaxGen    = 2
\*   WorkerIds = {1,2,3,4}

=============================================================================
