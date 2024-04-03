# Changelog

## Version 1.5.0b2

### New Features

- Added the `help()` function to provide a convenient way to explore the available methods in the NWP_Stats class and view their descriptions and usage instructions. Users can now easily access the documentation for specific methods or view the entire list of available methods. This enhancement addresses the issue #1 raised in the GitHub repository.

Example usage:
```python
from nwpeval import NWP_Stats

# Display help for a specific method
help(NWP_Stats.compute_fss)

# Display help for all available methods
help(NWP_Stats)
```

### Bug Fixes

- Fixed minor bugs and improved code stability.

### Other Changes

- The package has been moved from the 3-Alpha stage to the 4-Beta stage in development, indicating that it has undergone further testing and refinement.

Please note that this is a beta release (version 1.5.0b2), and while it includes significant enhancements and bug fixes, it may still have some known limitations or issues. We encourage users to provide feedback and report any bugs they encounter.

We appreciate your interest in the NWPeval package and thank you for your support!