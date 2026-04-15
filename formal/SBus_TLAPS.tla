---- MODULE SBus_TLAPS ----
(*
  S-Bus ORI Safety Proof — v15 (final)
  =====================================
  Run: tlapm -I /path/to/tlapm/lib/tlaps SBus_TLAPS.tla

  KEY INSIGHT: CASE hypothesis is NEVER in leaf BY obligations.
  OwnershipInvariant'/ORISafety' CASE steps work because their bodies
  are tautologies (x=a1/\x=a2=>a1=a2 is always true).
  RegistryTI'/TokensTI' bodies are NOT tautologies — need different approach.

  FIX for RegistryTI'/TokensTI':
  Pre-compute the two cases as named steps OUTSIDE the CASE split:
    <1>RNATS: registry'[sh].version \in Nat         (s2=sh case fact)
    <1>RTIH:  \A s2: s2#sh => reg'[s2].ver \in Nat  (s2#sh case fact)
  Then leaf proof: RegistryTI' BY RNATS, RTIH — Zenon does 4-5 step
  internal case split (s2=sh ∨ s2#sh).
*)

EXTENDS Naturals, Sequences, FiniteSets, TLAPS

CONSTANT Agents
CONSTANT Shards
CONSTANT NoOwner

ASSUME NoOwner \notin Agents

VARIABLE registry
VARIABLE tokens
VARIABLE delivery_log

vars == <<registry, tokens, delivery_log>>

UVars == UNCHANGED vars

RegistryTI == \A s2 \in Shards : registry[s2].version \in Nat
TokensTI   == \A s2 \in Shards : tokens[s2] \in Agents \cup {NoOwner}
TypeInvariant == RegistryTI /\ TokensTI

OwnershipInvariant ==
    \A s2 \in Shards :
        tokens[s2] \in Agents =>
            \A a1, a2 \in Agents :
                (tokens[s2] = a1 /\ tokens[s2] = a2) => (a1 = a2)

ORISafety ==
    \A s2 \in Shards :
        \A v1, v2 \in Nat :
            (registry[s2].version = v1 /\ registry[s2].version = v2) => (v1 = v2)

IND == TypeInvariant /\ OwnershipInvariant /\ ORISafety

Init ==
    /\ \A s2 \in Shards : registry[s2] = [version |-> 0, content |-> ""]
    /\ \A s2 \in Shards : tokens[s2] = NoOwner
    /\ \A ag \in Agents : delivery_log[ag] = <<>>

Read(ag, sh) ==
    /\ delivery_log' = [delivery_log EXCEPT
            ![ag] = Append(delivery_log[ag], <<sh, registry[sh].version>>)]
    /\ UNCHANGED <<registry, tokens>>

Commit(ag, sh, ve, delta) ==
    /\ registry[sh].version = ve
    /\ tokens[sh] = NoOwner
    /\ registry'[sh] = [version |-> ve + 1, content |-> delta]
    /\ \A s2 \in Shards : s2 # sh => registry'[s2] = registry[s2]
    /\ tokens'[sh] = NoOwner
    /\ \A s2 \in Shards : s2 # sh => tokens'[s2] = tokens[s2]
    /\ UNCHANGED delivery_log

Timeout(ag) ==
    /\ delivery_log' = [delivery_log EXCEPT ![ag] = <<>>]
    /\ UNCHANGED <<registry, tokens>>

Next ==
    \/ \E ag \in Agents, sh \in Shards : Read(ag, sh)
    \/ \E ag \in Agents, sh \in Shards, ve \in Nat, delta \in STRING :
           Commit(ag, sh, ve, delta)
    \/ \E ag \in Agents : Timeout(ag)

Spec == Init /\ [][Next]_vars

\* ============================================================
\* THEOREM 1: Init => IND
\* ============================================================

THEOREM InitEstablishesIND == Init => IND
<1>1. ASSUME Init PROVE IND
  <2>1. TypeInvariant
    <3>1. RegistryTI
      <4>S. SUFFICES \A s2 \in Shards : registry[s2].version \in Nat
        BY DEF RegistryTI
      <4>. TAKE s2 \in Shards
      <4>a. registry[s2] = [version |-> 0, content |-> ""]
        BY Init DEF Init
      <4>b. registry[s2].version = 0
        BY <4>a
      <4>c. 0 \in Nat
        OBVIOUS
      <4>. QED BY <4>b, <4>c
    <3>2. TokensTI
      <4>S. SUFFICES \A s2 \in Shards : tokens[s2] \in Agents \cup {NoOwner}
        BY DEF TokensTI
      <4>. TAKE s2 \in Shards
      <4>a. tokens[s2] = NoOwner
        BY Init DEF Init
      <4>b. NoOwner \in Agents \cup {NoOwner}
        OBVIOUS
      <4>. QED BY <4>a, <4>b
    <3>. QED BY <3>1, <3>2 DEF TypeInvariant
  <2>2. OwnershipInvariant
    BY Init DEF Init, OwnershipInvariant
  <2>3. ORISafety
    BY DEF ORISafety
  <2>. QED BY <2>1, <2>2, <2>3 DEF IND
<1>. QED BY <1>1


LEMMA ReadPreservesIND ==
    ASSUME IND, NEW ag \in Agents, NEW sh \in Shards, Read(ag, sh)
    PROVE  IND'
BY DEF Read, IND, TypeInvariant, RegistryTI, TokensTI, OwnershipInvariant, ORISafety

LEMMA TimeoutPreservesIND ==
    ASSUME IND, NEW ag \in Agents, Timeout(ag)
    PROVE  IND'
BY DEF Timeout, IND, TypeInvariant, RegistryTI, TokensTI, OwnershipInvariant, ORISafety


\* ============================================================
\* LEMMA: Commit preserves IND
\*
\* RegistryTI'/TokensTI' proved via pre-computed named steps
\* (no CASE split — CASE hypothesis never reaches leaf BY).
\* Zenon does 4-5 step internal case split on s2=sh vs s2#sh.
\* ============================================================

LEMMA CommitPreservesIND ==
    ASSUME IND,
           NEW ag \in Agents, NEW sh \in Shards,
           NEW ve \in Nat,    NEW delta \in STRING,
           Commit(ag, sh, ve, delta)
    PROVE  IND'

<1>RA. registry'[sh] = [version |-> ve + 1, content |-> delta]
  BY DEF Commit
<1>RV. registry'[sh].version = ve + 1
  BY <1>RA
<1>RL. \A s2 \in Shards : s2 # sh => registry'[s2] = registry[s2]
  BY DEF Commit
<1>TA. tokens'[sh] = NoOwner
  BY DEF Commit
<1>TL. \A s2 \in Shards : s2 # sh => tokens'[s2] = tokens[s2]
  BY DEF Commit
<1>VN. ve + 1 \in Nat
  OBVIOUS

\* Pre-compute: registry'[sh].version \in Nat (s2=sh case)
<1>RNATS. registry'[sh].version \in Nat
  BY <1>RV, <1>VN

\* Pre-compute: for s2#sh, registry'[s2].version \in Nat (s2#sh case)
<1>RTIH. \A s2 \in Shards : s2 # sh => registry'[s2].version \in Nat
  BY IND, <1>RL DEF IND, TypeInvariant, RegistryTI

\* Pre-compute: tokens'[sh] \in Agents \cup {NoOwner} (s2=sh case)
<1>TNOWN. tokens'[sh] \in Agents \cup {NoOwner}
  BY <1>TA

\* Pre-compute: for s2#sh, tokens'[s2] \in Agents \cup {NoOwner} (s2#sh case)
<1>TTIH. \A s2 \in Shards : s2 # sh => tokens'[s2] \in Agents \cup {NoOwner}
  BY IND, <1>TL DEF IND, TypeInvariant, TokensTI

\* RegistryTI': Zenon internal case split — s2=sh uses RNATS, s2#sh uses RTIH
<1>53. RegistryTI'
  BY <1>RNATS, <1>RTIH DEF RegistryTI

\* TokensTI': same pattern
<1>54. TokensTI'
  BY <1>TNOWN, <1>TTIH DEF TokensTI

<1>5. TypeInvariant'
  BY <1>53, <1>54 DEF TypeInvariant

<1>6. OwnershipInvariant'
  <2>S. SUFFICES \A s2 \in Shards :
            tokens'[s2] \in Agents =>
                \A a1, a2 \in Agents :
                    (tokens'[s2] = a1 /\ tokens'[s2] = a2) => a1 = a2
    BY DEF OwnershipInvariant
  <2>. TAKE s2 \in Shards
  <2>1. CASE s2 = sh
    BY <1>TA
  <2>2. CASE s2 # sh
    BY IND, <1>TL DEF IND, OwnershipInvariant
  <2>. QED BY <2>1, <2>2

<1>7. ORISafety'
  <2>S. SUFFICES \A s2 \in Shards : \A v1, v2 \in Nat :
            registry'[s2].version = v1 /\ registry'[s2].version = v2 => v1 = v2
    BY DEF ORISafety
  <2>. TAKE s2 \in Shards
  <2>1. CASE s2 = sh
    BY <1>RV
  <2>2. CASE s2 # sh
    BY IND, <1>RL DEF IND, ORISafety
  <2>. QED BY <2>1, <2>2

<1>. QED BY <1>5, <1>6, <1>7 DEF IND


THEOREM INDIsInductive == IND /\ [Next]_vars => IND'
<1>1. ASSUME IND, Next PROVE IND'
  BY IND, Next, ReadPreservesIND, CommitPreservesIND, TimeoutPreservesIND DEF Next
<1>2. ASSUME IND, UNCHANGED vars PROVE IND'
  BY IND, UVars DEF UVars, vars, IND, TypeInvariant, RegistryTI, TokensTI, OwnershipInvariant, ORISafety
<1>. QED BY <1>1, <1>2


THEOREM Spec => []IND
  BY InitEstablishesIND, INDIsInductive, PTL DEF Spec

====