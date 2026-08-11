"""Consecutive-frame pairs from the IK forward-walk dataset.

Each item is one transition: two augmented views of (frame_t, frame_t+1) plus the joint
command issued at t. Actions are standardised with statistics computed on the training
morphologies only, so a held-out body never influences the normalisation.
"""
import glob
import os

import numpy as np
from torch.utils.data import Dataset, Sampler

from .augment import apply, identity_params, sample_params
from .embodiment import REGISTRY
from .embodiment import load as load_embodiment

CONTACT_THRESHOLD = 0.27


def clip_paths(data_dir, morphs):
    # named explicitly here rather than globbed, so a non-walking body can only arrive by being
    # asked for; say so rather than silently obeying
    named = set(morphs) & set(EXCLUDED_BODIES)
    if named:
        print(f"WARNING: {sorted(named)} do not walk -- they collapse and rotate on the spot. "
              f"See FINDINGS.md F42.")
    paths = []
    for path in sorted(glob.glob(os.path.join(data_dir, "*.npz"))):
        if os.path.basename(path).split("_")[0] in morphs:
            paths.append(path)
    return paths


def available_episodes(data_dir, morphs):
    episodes = set()
    for path in clip_paths(data_dir, morphs):
        with np.load(path, allow_pickle=True) as data:
            episodes.add(int(data["expert_episode"]))
    return sorted(episodes)


def load_clip(path):
    with np.load(path, allow_pickle=True) as data:
        return {
            "frames": data["frames"],
            "actions": data["actions"].astype(np.float32),
            "forces": data["forces"].astype(np.float32),
            "morph": str(data["morph"]),
            "episode": int(data["expert_episode"]),
            "repeat": int(data["repeat"]),
        }


def action_stats(clips, within_body=True):
    """Per-joint mean and scale used to standardise the motion target.

    The scale decides how much each joint contributes to the motion loss, so pooling every
    body together is not neutral: bodies with different leg lengths hold their coxa-femur
    and femur-tibia joints at different mean angles, and that gap between bodies lands in
    the pooled standard deviation as if it were signal amplitude. Measured on
    data/ik_walk_100_framed with long and short as the training bodies, pooling gives a
    coxa-femur scale of 18.4 degrees while either body on its own moves that joint by only
    6.4-9.0 degrees, so the joint receives (6.4/18.4)^2 = 0.12 of the weight thorax-coxa
    gets. The joints that differ most between bodies are precisely the ones a
    cross-morphology model has to learn, so the pooled scale silences them.

    Averaging the per-body variances removes the between-body term and leaves each joint
    weighted by how much it actually moves. The mean stays pooled: an unseen body's mean
    posture is not knowable at test time.
    """
    actions = np.concatenate([clip["actions"] for clip in clips], axis=0)
    mean = actions.mean(axis=0)
    if within_body:
        groups = {}
        for clip in clips:
            groups.setdefault(clip["morph"], []).append(clip["actions"])
        variances = [np.concatenate(v, axis=0).var(axis=0) for v in groups.values()]
        std = np.sqrt(np.mean(variances, axis=0))
    else:
        std = actions.std(axis=0)
    return mean, np.maximum(std, 1e-6)


def contact_labels(forces):
    return (forces > CONTACT_THRESHOLD).astype(np.int64)


class IKWalkPairs(Dataset):
    """Split by expert episode, never by repeat: repeats of one episode share a bit-identical
    action sequence and near-identical frames, so holding one out measures nothing."""

    def __init__(self, data_dir, morphs, episodes=None, mean=None, std=None, seed=0,
                 frame_range=None, within_body_std=True, cross_augment=True, action_lag=1):
        self.clips = [load_clip(p) for p in clip_paths(data_dir, morphs)]
        if episodes is not None:
            keep = set(episodes)
            self.clips = [c for c in self.clips if c["episode"] in keep]
        if not self.clips:
            raise ValueError(f"no clips in {data_dir} for morphs={morphs} episodes={episodes}")

        if mean is None or std is None:
            mean, std = action_stats(self.clips, within_body_std)
        self.mean, self.std = mean.astype(np.float32), std.astype(np.float32)

        # stable body index for the adversarial head; sorted so a resumed or re-split run
        # assigns the same label to the same body
        self.morphs = sorted({clip["morph"] for clip in self.clips})
        self.morph_index = {name: i for i, name in enumerate(self.morphs)}

        # Every body walks the same expert episodes, so at a given episode and timestep all of
        # them are at the same point of the same shared Cartesian foot trajectory and differ only
        # in geometry. That makes it possible to decode one body's latent against another body's
        # frame and know what the answer should be, which is what cfg.lambda_cross trains on.
        self.partners = {}
        for i, clip in enumerate(self.clips):
            self.partners.setdefault(clip["episode"], {}).setdefault(clip["morph"], i)

        # The target sits action_lag steps past t, so a transition is only usable when that
        # index exists: t+1 for the frames and t+action_lag for the command.
        self.action_lag = action_lag
        reach = max(1, action_lag)
        start, stop = frame_range or (0, 0)
        self.index = [
            (i, t)
            for i, clip in enumerate(self.clips)
            for t in range(start, (stop or len(clip["frames"])) - reach)
        ]
        self.seed = seed
        self.epoch = 0
        # False gives the ITM and FTM the same un-augmented frames. What stops the ITM copying
        # x_{t+1} into z is then the dimensional bottleneck alone: z is 64 numbers against
        # e_{t+1}'s 359,000. See FINDINGS.md F25 for why the augmentation had to go.
        self.cross_augment = cross_augment

    def set_epoch(self, epoch):
        """Fresh augmentations each epoch while keeping the run reproducible."""
        self.epoch = epoch

    def __len__(self):
        return len(self.index)

    def __getitem__(self, i):
        clip_idx, t = self.index[i]
        clip = self.clips[clip_idx]
        frame_t, frame_next = clip["frames"][t], clip["frames"][t + 1]

        rng = np.random.default_rng((self.seed, self.epoch, i))
        height, width = frame_t.shape[:2]
        if self.cross_augment:
            a1 = sample_params(rng, height, width)
            a2 = sample_params(rng, height, width)
        else:
            a1 = a2 = identity_params(height, width)

        action = (clip["actions"][t + self.action_lag] - self.mean) / self.std
        sample = {
            "view1_t": apply(frame_t, a1),
            "view1_next": apply(frame_next, a1),
            "view2_t": apply(frame_t, a2),
            "view2_next": apply(frame_next, a2),
            "action": action.astype(np.float32),
            "morph_id": self.morph_index[clip["morph"]],
        }

        # a different body at the same episode and timestep: same intent, different geometry.
        # Its own augmentation, so the two frames share no nuisance factor the model could match
        # on instead of reading the body.
        others = [m for m in self.partners[clip["episode"]] if m != clip["morph"]]
        if others:
            partner = self.clips[self.partners[clip["episode"]][others[rng.integers(len(others))]]]
            a3 = sample_params(rng, height, width) if self.cross_augment \
                else identity_params(height, width)
            sample["cross_x_t"] = apply(partner["frames"][t], a3)
            sample["cross_action"] = (
                (partner["actions"][t + self.action_lag] - self.mean) / self.std).astype(np.float32)
            sample["cross_morph_id"] = self.morph_index[partner["morph"]]
        return sample


# Bodies whose recorded episodes show them collapsing rather than walking. A two-link leg cannot
# reach closer to its shoulder than |femur - tibia|, and the closest commanded target sits at
# 92.5 mm; both of these have a 208.2 mm dead zone, so the IK never solves and the insect tips onto
# its side within about a dozen frames and rotates on the spot for the rest of the episode.
#
# They predate `sim/make_leg_morphology.py`'s reach check -- which was written using them as its
# evidence -- and survived the walk check because it measured `norm(head[-1,:2] - head[0,:2])`,
# unsigned, which reads a healthy 0.46 m for a body that is tumbling. Excluded by name because
# Stage 2 takes a directory and globs it, so nothing else selects bodies at all.
#
# This is not a judgement about morphology. Every other body in the set locomotes; these two are
# recordings of a robot falling over. See FINDINGS.md F42.
EXCLUDED_BODIES = ("c06f06t10", "c10f06t10")


def usable_clips(paths, excluded=EXCLUDED_BODIES):
    """Drop clips belonging to bodies that do not walk, reporting what went."""
    kept = [p for p in paths if os.path.basename(p).split("_")[0] not in excluded]
    dropped = len(paths) - len(kept)
    if dropped:
        names = sorted({os.path.basename(p).split("_")[0] for p in paths}
                       & set(excluded))
        print(f"excluding {dropped} clips from non-walking bodies {names} "
              f"({len(kept)} clips remain)")
    return kept


def evenly(items, keep):
    """`keep` items spread across the list, not the first `keep`.

    Clips are sorted by episode, so taking a prefix would take consecutive expert episodes and
    the behavioural range would narrow along with the count.
    """
    if keep >= len(items):
        return items
    idx = np.linspace(0, len(items) - 1, keep).round().astype(int)
    return [items[i] for i in dict.fromkeys(idx)]


def embodiment_split(specs, val_fraction, root="", exclude=True, heldout_bodies=(),
                     clips_per_body=()):
    """Split each embodiment's clips into train and validation source lists.

    Splitting is by clip rather than by expert episode because embodiments other than the
    hexapod have no episode structure to split on. The held-out amount is a fraction, not a
    count: embodiments differ by orders of magnitude in clip count, so a fixed count would
    take a negligible slice from one and half the data from another.

    **Stratified by body.** An earlier version took the last `val_fraction` of the sorted path
    list, and sorted paths group by body, so the tail was one body: validation consisted entirely
    of `c10f10t10` while that same body kept only 12 of its 30 clips for training. Every Stage 2
    run before 2026-08-11 has that split -- one body under-represented in training and the
    validation metric blind to the other five. Taking the fraction from each body separately
    fixes both.
    """
    caps = dict(s.split("=") for s in clips_per_body) if clips_per_body else {}
    caps = {k: int(v) for k, v in caps.items()}
    train, val = [], []
    for name, data_dir in specs:
        directory = data_dir if os.path.isabs(data_dir) else os.path.join(root, data_dir)
        paths = sorted(glob.glob(os.path.join(directory, "*.npz")))
        if exclude:
            paths = usable_clips(paths)
        if heldout_bodies:
            # A body kept out of training entirely, so cross-embodiment training has at least one
            # generalisation test. Without it Stage 2 globs every body and can only measure what
            # is *inside* the latent, never whether anything transfers -- the two embodiments
            # cannot serve as each other's held-out set, since holding one out leaves one.
            kept = [p for p in paths
                    if os.path.basename(p).split("_")[0] not in heldout_bodies]
            if len(kept) < len(paths):
                print(f"{name}: holding out {sorted(set(heldout_bodies))} "
                      f"({len(paths) - len(kept)} clips withheld from training)")
            paths = kept
        if not val_fraction:
            train.append((paths, name))
            continue

        by_body = {}
        for p in paths:
            by_body.setdefault(os.path.basename(p).split("_")[0], []).append(p)
        if name in caps:
            before = sum(len(v) for v in by_body.values())
            by_body = {b: evenly(sorted(c), caps[name]) for b, c in by_body.items()}
            after = sum(len(v) for v in by_body.values())
            print(f"{name}: capped at {caps[name]} clips per body, {before} -> {after} clips")
        tr, va = [], []
        for body, clips in sorted(by_body.items()):
            n_val = max(1, round(len(clips) * val_fraction))
            if len(clips) - n_val < 1:
                raise ValueError(f"{name}/{body}: {len(clips)} clips is too few to split")
            tr.extend(clips[:len(clips) - n_val])
            va.extend(clips[len(clips) - n_val:])
        if not tr:
            raise ValueError(f"{name}: {len(paths)} clips is too few to split")
        train.append((tr, name))
        if va:
            val.append((va, name))
            print(f"{name}: {len(tr)} train / {len(va)} val clips, "
                  f"stratified over {len(by_body)} bod{'y' if len(by_body) == 1 else 'ies'}")
    return train, val


class MultiEmbodimentPairs(Dataset):
    """Transitions pooled across embodiments with different action dimensionalities.

    Actions are standardised per embodiment, since 18-D insect joint targets and 12-D
    quadruped joint targets share no units or correspondence.
    """

    def __init__(self, sources, stats=None, seed=0, cross_augment=True, action_lag=1):
        self.clips, self.stats = [], {}
        for paths, name in sources:
            spec = REGISTRY[name]
            clips = [load_embodiment(p, spec) for p in paths]
            if not clips:
                raise ValueError(f"no clips given for embodiment {name}")
            if stats and name in stats:
                self.stats[name] = stats[name]
            else:
                actions = np.concatenate([c["actions"] for c in clips])
                self.stats[name] = (actions.mean(0), np.maximum(actions.std(0), 1e-6))
            self.clips.extend(clips)

        self.action_lag = action_lag
        reach = max(1, action_lag)
        self.index = [
            (i, t)
            for i, clip in enumerate(self.clips)
            for t in range(len(clip["frames"]) - reach)
        ]
        self.seed = seed
        self.epoch = 0
        # False gives the ITM and FTM the same un-augmented frames. What stops the ITM copying
        # x_{t+1} into z is then the dimensional bottleneck alone: z is 64 numbers against
        # e_{t+1}'s 359,000. See FINDINGS.md F25 for why the augmentation had to go.
        self.cross_augment = cross_augment

    def set_epoch(self, epoch):
        self.epoch = epoch

    def embodiment_indices(self):
        groups = {}
        for i, (clip_idx, _) in enumerate(self.index):
            groups.setdefault(self.clips[clip_idx]["embodiment"], []).append(i)
        return groups

    def frame_at(self, i):
        """The raw frame for pair `i`, with no augmentation applied.

        Used to estimate the per-embodiment appearance offset, which is a property of how a robot
        renders and should not be measured through a random crop.
        """
        clip_idx, t = self.index[i]
        return self.clips[clip_idx]["frames"][t]

    def __len__(self):
        return len(self.index)

    def __getitem__(self, i):
        clip_idx, t = self.index[i]
        clip = self.clips[clip_idx]
        frame_t, frame_next = clip["frames"][t], clip["frames"][t + 1]

        rng = np.random.default_rng((self.seed, self.epoch, i))
        height, width = frame_t.shape[:2]
        if self.cross_augment:
            a1 = sample_params(rng, height, width)
            a2 = sample_params(rng, height, width)
        else:
            a1 = a2 = identity_params(height, width)

        mean, std = self.stats[clip["embodiment"]]
        return {
            "view1_t": apply(frame_t, a1),
            "view1_next": apply(frame_next, a1),
            "view2_t": apply(frame_t, a2),
            "view2_next": apply(frame_next, a2),
            "action": ((clip["actions"][t + self.action_lag] - mean) / std).astype(np.float32),
            "embodiment": clip["embodiment"],
        }


class EmbodimentBatchSampler(Sampler):
    """One embodiment per batch, so action tensors stay rectangular.

    `balance` decides how much of each epoch each embodiment gets. Proportional draws batches in
    proportion to dataset size, which on the insect-plus-B1 pairing gives the quadruped 1,062 of
    16,817 pairs -- 6.3 percent of the gradient steps. A shared trunk trained that way is a
    hexapod model with a quadruped footnote, and any claim about a *shared* latent space would be
    confounded by the imbalance rather than measuring it.

    Balanced repeats the smaller embodiment until every embodiment contributes the same number of
    batches. The cost is that each B1 transition is seen roughly fifteen times per epoch, so B1
    overfitting has to be watched rather than assumed away.
    """

    def __init__(self, dataset, batch_size, shuffle=True, seed=0, balance=True):
        self.groups = dataset.embodiment_indices()
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.seed = seed
        self.balance = balance
        self.epoch = 0

    def set_epoch(self, epoch):
        self.epoch = epoch

    def _batches_per_group(self):
        counts = {name: -(-len(v) // self.batch_size) for name, v in self.groups.items()}
        if self.balance:
            most = max(counts.values())
            return {name: most for name in counts}
        return counts

    def __len__(self):
        return sum(self._batches_per_group().values())

    def __iter__(self):
        rng = np.random.default_rng((self.seed, self.epoch))
        wanted = self._batches_per_group()
        batches = []
        for name, indices in self.groups.items():
            need = wanted[name] * self.batch_size
            order = []
            while len(order) < need:
                pool = rng.permutation(indices) if self.shuffle else np.array(indices)
                order.extend(pool.tolist())
            order = np.array(order[:need])
            batches += [order[i:i + self.batch_size].tolist()
                        for i in range(0, len(order), self.batch_size)]
        if self.shuffle:
            batches = [batches[i] for i in rng.permutation(len(batches))]
        return iter(batches)


class IKWalkFrames(Dataset):
    """Un-augmented frames for evaluation and probing."""

    def __init__(self, data_dir, morphs, mean=None, std=None, action_lag=1):
        self.clips = [load_clip(p) for p in clip_paths(data_dir, morphs)]
        if mean is None or std is None:
            mean, std = action_stats(self.clips)
        self.mean, self.std = mean.astype(np.float32), std.astype(np.float32)
        self.action_lag = action_lag
        reach = max(1, action_lag)
        self.index = [
            (i, t)
            for i, clip in enumerate(self.clips)
            for t in range(len(clip["frames"]) - reach)
        ]

    def __len__(self):
        return len(self.index)

    def __getitem__(self, i):
        clip_idx, t = self.index[i]
        clip = self.clips[clip_idx]
        action = (clip["actions"][t + self.action_lag] - self.mean) / self.std
        return {
            "frame_t": clip["frames"][t],
            "frame_next": clip["frames"][t + 1],
            "action": action.astype(np.float32),
            "raw_action": clip["actions"][t + self.action_lag],
            "contact": contact_labels(clip["forces"][t]),
            "morph": clip["morph"],
            "episode": clip["episode"],
            "repeat": clip["repeat"],
        }
