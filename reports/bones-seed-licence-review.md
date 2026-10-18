# BONES-SEED licence: clause map for the published ternary bundle

Status: engineering analysis for legal review. This is not legal advice, and it does not resolve the licence question.

Sources: https://bones.studio/info/seed-license (fetched; clauses quoted verbatim from the page).
Artifact: `model/ternary/` (pose, root and VQ-VAE decoder weights, derived from BONES-SEED features).

## Clauses that apply

**Results (Section 5.1).** "Any publicly or privately distributed output created by Licensee that is based upon or incorporates the Dataset, such as trained machine learning models ... Results include any derivative works based on the Dataset."

**Derivative permission (Section 5.2 as quoted).** "Licensee acquires all proprietary copyrights to any Results ... This permission is valid solely provided that the raw Dataset itself cannot be extracted, reverse-engineered, or re-obtained from such Results."

**No Competing Datasets & Generative Models.** "Use the Dataset or Results to create, improve, or augment a Competing Dataset or a commercially substitutable synthetic dataset. This includes ... using the Dataset to train, fine-tune, or condition any generative model whose primary purpose or substantial use is to generate motion capture data, human motion sequences, animation data ... For the avoidance of doubt, this restriction does not prohibit training machine learning models (such as robotic control policies ... or motion tracking models) that consume the Dataset as training data and produce non-data outputs."

**Eligibility.** "Academic User: a non-profit academic institution ... using the Dataset strictly for non-commercial scientific research that is intended to be and is subsequently made publicly available." "Qualifying Startup: a commercial entity which, at the time of accessing the Dataset has an annual gross revenue of less than $1,000,000 USD."

**Termination.** On breach of a material term the licence ends, and the Licensee must cease use of Results and withdraw them from products or services. Section 5.2 permission is revoked automatically on termination.

## How the bundle maps to these clauses

| Question | What the record shows | Open for counsel |
|---|---|---|
| Are the weights "Results"? | Yes on the plain wording: trained models derived from the Dataset. | None: the clause names trained models. |
| Can the raw dataset be recovered from the weights? | Not tested. The weights are ternary/Q8/F32 derived from ~9,600 to 109,600 clips for 3,000 steps. | Needs a memorisation or extraction test before publication is defensible. |
| Is the bundle a "generative model whose primary purpose ... is to generate human motion sequences"? | The bundle is a motion-generation model (MotionBricks G1 pose, root and decoder). It produces motion sequences from conditioning. | Whether its primary purpose is generation is a legal question; the clause carves out models that produce non-data outputs, and this one produces motion. |
| Could its output "function as a commercial substitute"? | The page emits short generated clips. | Substitution is defined by "could reasonably be licensed ... as a replacement for the Dataset". |
| Who is publishing? | The repo is AnEntrypoint/mbweb (public). | The licensee's status (Academic User, Qualifying Startup, or neither) is not recorded here. |

## Decisions already taken

- Owner determination (recorded in the conversation): the weights are licence-compatible. Stated reasons: the bundle is a quantization of the same model, so it does not compete with the dataset; and it is trained on the same source set, so it introduces no new licence terms. This is the owner's reading, not an external legal review.

- The owner decided to publish the weights after being shown the clauses above (recorded in the conversation).
- No engineering step removes the risk. Options open to the owner: keep the weights out of the public repo, publish code only, obtain a written licence from bones.studio/licensing@bones.studio, or withdraw the weights if counsel advises against it.

## Facts counsel needs

1. The publishing entity's status under the licence (Academic User, Qualifying Startup, or other).
2. Whether any BONES-SEED clips appear in the weights in recoverable form (requires the test in row 2).
3. Whether a written licence or waiver from Bones Studio is available.
4. Whether the public repo's distribution of derived weights is "publicly distributed output" under Section 5.1 and remains within the permission.
