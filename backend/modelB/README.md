# Model B — HandLex (Archived Checkpoints)

## Current status

Model B currently has archived checkpoint and configuration files in the backend repository. The archived files alone do not establish that a complete, runnable inference pipeline has been integrated into HandLex.

## Archived checkpoint groups

The archive contains checkpoint/configuration pairs associated with NSLT experiments for:

- ASL100
- ASL300
- ASL1000
- ASL2000

The filenames include experiment iteration and reported Top-1, Top-5, and Top-10 values. These values are checkpoint-specific metadata and should not be compared directly with Model A's benchmark without considering dataset, task, evaluation protocol, and preprocessing.

## Integration status

- [x] Checkpoint files archived
- [x] Configuration files archived
- [ ] Runtime implementation integrated
- [ ] Preprocessing pipeline verified
- [ ] Inference API implemented
- [ ] Runtime testing performed in HandLex

## Dependencies

The active Model B runtime and expected preprocessing have not yet been established in the current integration. Add and pin PyTorch/I3D or other required dependencies only after the exact implementation is identified and tested.

## Future work

1. Identify and document the original architecture and source.
2. Verify checkpoint/config compatibility.
3. Document preprocessing and input requirements.
4. Implement isolated inference.
5. Evaluate using a reproducible protocol.
6. Integrate with the API only after validation.

Do not treat archived checkpoints as a currently functioning HandLex model.
