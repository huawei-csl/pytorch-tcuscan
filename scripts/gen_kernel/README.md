## Contributing a new kernel

To develop a new AscendC kernel that follows the project conventions, you can use the `gen_new_kernel` tool.

```bash
# Adjust the Makefile with your new kernel name, i.e., `TCUSCAN_KERNEL` and `CPP_NAMESPACE` variables.

# Codegen (for debugging)
make codegen

# Codegen and deploy
make deploy
```

Plese don't forget to add a unit test in the CI with your new kernel.

### Test locally

To make sure the kernel has been integrated successfully, type `make test_awesome_kernel` where `awesome_kernel` is the name of the kernel specified in `TCUSCAN_KERNEL` variable.
