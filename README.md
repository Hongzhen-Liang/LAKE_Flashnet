# Instructions for compiling kernel

Start with Ubuntu 20 or 22. We assume gcc is installed.

## Install dependencies

```
sudo apt-get update
sudo apt-get -y install build-essential tmux git pkg-config cmake zsh
sudo apt-get install libncurses-dev gawk flex bison openssl libssl-dev dkms libelf-dev libiberty-dev autoconf zstd
sudo apt-get install libreadline-dev binutils-dev libnl-3-dev
sudo apt-get install ecryptfs-utils cpufrequtils 
```

## Compile kernel

Clone `https://github.com/utcs-scea/LAKE-linux-6.0.git`.
Go in the directory and run `full_compilation.sh`, it should do everything.

If you are running with a monitor, reboot and choose the new kernel in grub.
Otherwise, make the new kernel the default by:
1. Open `/boot/grub/grub.cfg`, scroll down until you see boot options, then write down the id for the advanced menu and the id for the 6.0-lake.
You can use `cat /boot/grub/grub.cfg | grep submenu` and `cat /boot/grub/grub.cfg | grep option | grep 6.0.0-lake` to get the id for the advanced menu and  the 6.0-lake respectively.
3. Join them (advanced menu and kernel id), in that order with a `>`. For example:
`gnulinux-advanced-11b57fec-e05f-4c4d-8d80-445381841fa1>gnulinux-6.0-hack-advanced-11b57fec-e05f-4c4d-8d80-445381841fa1`
3. Open `/etc/default/grub` and, at the top of the file, add a default option using the string above. For example:
`GRUB_DEFAULT="gnulinux-advanced-11b57fec-e05f-4c4d-8d80-445381841fa1>gnulinux-5.15.68-hack-advanced-11b57fec-e05f-4c4d-8d80-445381841fa1"`
4. Add to `GRUB_CMDLINE_LINUX_DEFAULT` (create if it doesn't exist): `cma=128M@0-4G log_buf_len=16M`
For example: `GRUB_CMDLINE_LINUX_DEFAULT="quiet splash cma=128M@0-4G log_buf_len=16M"`
5. Finally, run `sudo update-grub`. Reboot and make sure the lake kernel is right by running `uname -r`



# Basic Test

The `basic_test.sh` script runs the hello module and echo success if all the steps were successful. It also prints the steps to be taken in case running LAKE system fails. To run it, just do `./basic_test.sh` from the home directory of LAKE.

