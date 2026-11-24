"""Owned Windows subprocess family for a private benchmark runner.

Lazy Win32 loading: import/help/planning remain possible on other platforms.
No administrator token, privilege changes, breakaway, or security workaround.
The executable is created SUSPENDED and assigned at creation via JOB_LIST.
Only selected duplicated stdio handles are inherited; never the job handle.
"""
from __future__ import annotations

import ctypes as C
from ctypes import wintypes as W
import os
from pathlib import Path
import subprocess
import time

SIZE_T=C.c_size_t
ULONG_PTR=C.c_size_t
LONGLONG=C.c_longlong
ULONGLONG=C.c_ulonglong
HANDLE=W.HANDLE


class STARTUPINFOW(C.Structure):
    _fields_=[('cb',W.DWORD),('lpReserved',W.LPWSTR),('lpDesktop',W.LPWSTR),('lpTitle',W.LPWSTR),
        ('dwX',W.DWORD),('dwY',W.DWORD),('dwXSize',W.DWORD),('dwYSize',W.DWORD),
        ('dwXCountChars',W.DWORD),('dwYCountChars',W.DWORD),('dwFillAttribute',W.DWORD),
        ('dwFlags',W.DWORD),('wShowWindow',W.WORD),('cbReserved2',W.WORD),('lpReserved2',C.c_void_p),
        ('hStdInput',HANDLE),('hStdOutput',HANDLE),('hStdError',HANDLE)]


class STARTUPINFOEXW(C.Structure):
    _fields_=[('StartupInfo',STARTUPINFOW),('lpAttributeList',C.c_void_p)]


class PROCESS_INFORMATION(C.Structure):
    _fields_=[('hProcess',HANDLE),('hThread',HANDLE),('dwProcessId',W.DWORD),('dwThreadId',W.DWORD)]


class BASIC_LIMIT(C.Structure):
    _fields_=[('PerProcessUserTimeLimit',LONGLONG),('PerJobUserTimeLimit',LONGLONG),
        ('LimitFlags',W.DWORD),('MinimumWorkingSetSize',SIZE_T),('MaximumWorkingSetSize',SIZE_T),
        ('ActiveProcessLimit',W.DWORD),('Affinity',ULONG_PTR),('PriorityClass',W.DWORD),('SchedulingClass',W.DWORD)]


class IO_COUNTERS(C.Structure):
    _fields_=[(x,ULONGLONG) for x in ['ReadOperationCount','WriteOperationCount','OtherOperationCount',
        'ReadTransferCount','WriteTransferCount','OtherTransferCount']]


class EXTENDED_LIMIT(C.Structure):
    _fields_=[('BasicLimitInformation',BASIC_LIMIT),('IoInfo',IO_COUNTERS),
        ('ProcessMemoryLimit',SIZE_T),('JobMemoryLimit',SIZE_T),('PeakProcessMemoryUsed',SIZE_T),('PeakJobMemoryUsed',SIZE_T)]


class BASIC_ACCOUNTING(C.Structure):
    _fields_=[('TotalUserTime',LONGLONG),('TotalKernelTime',LONGLONG),
        ('ThisPeriodTotalUserTime',LONGLONG),('ThisPeriodTotalKernelTime',LONGLONG),
        ('TotalPageFaultCount',W.DWORD),('TotalProcesses',W.DWORD),('ActiveProcesses',W.DWORD),
        ('TotalTerminatedProcesses',W.DWORD)]


class WinAPI:
    def __init__(self):
        if os.name!='nt':raise NotImplementedError('Windows Job Object execution is qualified only on Windows10+')
        self.k=C.WinDLL('kernel32',use_last_error=True)
        specs={
            'CreateJobObjectW':([C.c_void_p,W.LPCWSTR],HANDLE),
            'SetInformationJobObject':([HANDLE,C.c_int,C.c_void_p,W.DWORD],W.BOOL),
            'QueryInformationJobObject':([HANDLE,C.c_int,C.c_void_p,W.DWORD,C.POINTER(W.DWORD)],W.BOOL),
            'IsProcessInJob':([HANDLE,HANDLE,C.POINTER(W.BOOL)],W.BOOL),
            'TerminateJobObject':([HANDLE,W.UINT],W.BOOL),
            'CloseHandle':([HANDLE],W.BOOL),
            'GetCurrentProcess':([],HANDLE),
            'GetStdHandle':([W.DWORD],HANDLE),
            'DuplicateHandle':([HANDLE,HANDLE,HANDLE,C.POINTER(HANDLE),W.DWORD,W.BOOL,W.DWORD],W.BOOL),
            'InitializeProcThreadAttributeList':([C.c_void_p,W.DWORD,W.DWORD,C.POINTER(SIZE_T)],W.BOOL),
            'UpdateProcThreadAttribute':([C.c_void_p,W.DWORD,ULONG_PTR,C.c_void_p,SIZE_T,C.c_void_p,C.c_void_p],W.BOOL),
            'DeleteProcThreadAttributeList':([C.c_void_p],None),
            'CreateProcessW':([W.LPCWSTR,W.LPWSTR,C.c_void_p,C.c_void_p,W.BOOL,W.DWORD,C.c_void_p,W.LPCWSTR,C.c_void_p,C.POINTER(PROCESS_INFORMATION)],W.BOOL),
            'ResumeThread':([HANDLE],W.DWORD),
            'WaitForSingleObject':([HANDLE,W.DWORD],W.DWORD),
            'GetExitCodeProcess':([HANDLE,C.POINTER(W.DWORD)],W.BOOL),
            'TerminateProcess':([HANDLE,W.UINT],W.BOOL),
            'GetProcessTimes':([HANDLE,C.POINTER(W.FILETIME),C.POINTER(W.FILETIME),C.POINTER(W.FILETIME),C.POINTER(W.FILETIME)],W.BOOL),
        }
        for name,(args,result) in specs.items():
            fn=getattr(self.k,name);fn.argtypes=args;fn.restype=result

    def ok(self,value,what):
        if not value:raise C.WinError(C.get_last_error(),what)
        return value


class OwnedPopen:
    """Small Popen-style adapter; returns suspended until explicit resume().

    registry(owner) runs before creating any executable, even if __init__ later
    raises. poll()/wait() require the ENTIRE job family to exit. root_poll()
    separately exposes the direct launcher status. close() is idempotent and
    kills/reaps any remaining owned family. Only CreateProcess descendants are
    covered; no breakaway or out-of-job WMI/service creation is supported.
    """
    def __init__(self,args,*,stdin=None,stdout=None,stderr=None,cwd=None,env=None,
                 text=False,universal_newlines=None,encoding=None,errors=None,bufsize=-1,
                 registry=None,resume=False,fault=None,creationflags=0,close_fds=True,shell=False,**unsupported):
        self.args=list(args);self.pid=None;self.creation_time=None;self.creation_time_100ns=None
        self.returncode=None;self.root_returncode=None;self.stdin=self.stdout=self.stderr=None
        self._closed=False;self._job=None;self._pi=PROCESS_INFORMATION();self._streams=[]
        self._pending_fds=[];self._temp_files=[];self._duplicates=[];self._thread_resumed=False
        self.cleanup_receipt=None;self.stages=[];self._fault=fault
        self._api=WinAPI()
        if unsupported:raise TypeError('Unsupported OwnedPopen options: '+','.join(unsupported))
        if creationflags & ~0x08000000:raise ValueError('Extra creation flags forbidden; only NO_WINDOW is compatible')
        if not close_fds or shell:raise ValueError('Explicit stdio handle list and no shell required')
        if not self.args or any(not isinstance(x,str) or '\0' in x for x in self.args):raise ValueError('Explicit string argv required')
        executable=Path(self.args[0]).resolve()
        if not executable.is_file():raise FileNotFoundError(executable)
        self.args[0]=str(executable)
        text=bool(text or universal_newlines or encoding or errors)
        try:
            self._job=self._api.ok(self._api.k.CreateJobObjectW(None,None),'CreateJobObjectW')
            limit=EXTENDED_LIMIT();limit.BasicLimitInformation.LimitFlags=0x2000
            self._api.ok(self._api.k.SetInformationJobObject(self._job,9,C.byref(limit),C.sizeof(limit)),'KILL_ON_JOB_CLOSE')
            self.stages.append('JOB_CREATED_KILL_ON_CLOSE_NO_BREAKAWAY')
            if registry is not None:registry(self)
            self.stages.append('REGISTERED_BEFORE_CREATE')
            handles=self._stdio(stdin,stdout,stderr,text,encoding or 'utf-8',errors or 'strict',bufsize)
            self._create(executable,handles,cwd,env)
            self.stages.append('CREATED_SUSPENDED_ASSIGNED_AT_CREATION')
            if self._fault:self._fault('after_create',self)
            self._member()
            if self.query()['ActiveProcesses']!=1:raise RuntimeError('Suspended root must be sole job member before resume')
            self.stages.append('MEMBERSHIP_VERIFIED_BEFORE_RESUME')
            if resume:self.resume()
        except BaseException:
            self.close()
            raise

    def _duplicate(self,handle):
        result=HANDLE();current=self._api.k.GetCurrentProcess()
        self._api.ok(self._api.k.DuplicateHandle(current,HANDLE(handle),current,C.byref(result),0,True,2),'Duplicate stdio handle')
        self._duplicates.append(result);return result.value

    def _stdio(self,stdin,stdout,stderr,text,encoding,errors,bufsize):
        import msvcrt
        descriptors=[]
        for index,value in enumerate([stdin,stdout,stderr]):
            if value==subprocess.STDOUT:
                if index!=2:raise ValueError('STDOUT redirection valid only for stderr')
                descriptors.append(descriptors[1]);continue
            if value==subprocess.PIPE:
                r,w=os.pipe();child_fd,parent_fd=(r,w) if index==0 else (w,r)
                self._pending_fds.append(child_fd)
                mode=('w' if index==0 else 'r')+('' if text else 'b')
                buffering=bufsize if text else (-1 if bufsize==1 else bufsize)
                stream=os.fdopen(parent_fd,mode,buffering=buffering,
                                 **(dict(encoding=encoding,errors=errors) if text else {}))
                self._streams.append(stream);setattr(self,['stdin','stdout','stderr'][index],stream)
                raw=msvcrt.get_osfhandle(child_fd)
            elif value==subprocess.DEVNULL:
                stream=open(os.devnull,'rb' if index==0 else 'wb');self._temp_files.append(stream);raw=msvcrt.get_osfhandle(stream.fileno())
            elif value is None:
                raw=self._api.k.GetStdHandle(W.DWORD((-10-index)&0xffffffff))
                if raw in (None,0,C.c_void_p(-1).value):
                    stream=open(os.devnull,'rb' if index==0 else 'wb');self._temp_files.append(stream);raw=msvcrt.get_osfhandle(stream.fileno())
            else:raw=msvcrt.get_osfhandle(value if isinstance(value,int) else value.fileno())
            descriptors.append(self._duplicate(raw))
        return descriptors

    def _release_setup(self):
        for value in self._duplicates:self._api.k.CloseHandle(value)
        self._duplicates.clear()
        for fd in self._pending_fds:
            try:os.close(fd)
            except OSError:pass
        self._pending_fds.clear()
        for stream in self._temp_files:stream.close()
        self._temp_files.clear()

    def _create(self,executable,handles,cwd,env):
        size=SIZE_T();self._api.k.InitializeProcThreadAttributeList(None,2,0,C.byref(size))
        if not size.value:raise C.WinError(C.get_last_error(),'Attribute list size')
        storage=C.create_string_buffer(size.value);attrs=C.cast(storage,C.c_void_p)
        self._api.ok(self._api.k.InitializeProcThreadAttributeList(attrs,2,0,C.byref(size)),'Initialize attribute list')
        job_array=(HANDLE*1)(self._job)
        unique=list(dict.fromkeys(handles));handle_array=(HANDLE*len(unique))(*unique)
        try:
            # SDK ProcThreadAttributeValue(13,FALSE,TRUE,FALSE), and HANDLE_LIST2.
            self._api.ok(self._api.k.UpdateProcThreadAttribute(attrs,0,0x2000D,job_array,C.sizeof(job_array),None,None),'Creation-time JOB_LIST')
            self._api.ok(self._api.k.UpdateProcThreadAttribute(attrs,0,0x20002,handle_array,C.sizeof(handle_array),None,None),'Explicit HANDLE_LIST')
            info=STARTUPINFOEXW();info.StartupInfo.cb=C.sizeof(info);info.lpAttributeList=attrs
            info.StartupInfo.dwFlags=0x100;info.StartupInfo.hStdInput,info.StartupInfo.hStdOutput,info.StartupInfo.hStdError=handles
            command=C.create_unicode_buffer(subprocess.list2cmdline(self.args))
            environment=None
            if env is not None:
                if any(not isinstance(k,str) or not isinstance(v,str) or '=' in k or '\0' in k+v for k,v in env.items()):raise ValueError('Invalid Unicode environment')
                environment=C.create_unicode_buffer('\0'.join(k+'='+v for k,v in sorted(env.items(),key=lambda kv:kv[0].upper()))+'\0\0')
            flags=0x4|0x400|0x80000|0x08000000
            self._api.ok(self._api.k.CreateProcessW(str(executable),command,None,None,True,flags,
                         environment,str(Path(cwd).resolve()) if cwd else None,C.byref(info),C.byref(self._pi)),'CreateProcessW SUSPENDED + JOB_LIST')
            self.pid=int(self._pi.dwProcessId)
            creation,exit_time,kernel,user=(W.FILETIME() for _ in range(4))
            self._api.ok(self._api.k.GetProcessTimes(self._pi.hProcess,C.byref(creation),C.byref(exit_time),C.byref(kernel),C.byref(user)),'Handle-bound creation time')
            self.creation_time_100ns=(creation.dwHighDateTime<<32)|creation.dwLowDateTime
            self.creation_time=self.creation_time_100ns/10**7-11644473600
        finally:
            self._api.k.DeleteProcThreadAttributeList(attrs)
            self._release_setup()

    def _member(self):
        flag=W.BOOL();self._api.ok(self._api.k.IsProcessInJob(self._pi.hProcess,self._job,C.byref(flag)),'IsProcessInJob')
        if not flag.value:raise RuntimeError('Owned root is not in its job; no resume')

    def _raw_query(self):
        data=BASIC_ACCOUNTING();self._api.ok(self._api.k.QueryInformationJobObject(self._job,1,C.byref(data),C.sizeof(data),None),'Job accounting')
        return dict(ActiveProcesses=int(data.ActiveProcesses),TotalProcesses=int(data.TotalProcesses),
                    TotalTerminatedProcesses=int(data.TotalTerminatedProcesses),CPU_s=(data.TotalUserTime+data.TotalKernelTime)/10**7)

    def query(self):
        if self._closed:return dict(self.cleanup_receipt['final_job_accounting'])
        return self._raw_query()

    def active_pids(self):
        if self._closed:return []
        count=16
        while count<=4096:
            raw=C.create_string_buffer(8+C.sizeof(ULONG_PTR)*count)
            if self._api.k.QueryInformationJobObject(self._job,3,raw,C.sizeof(raw),None):
                found=C.cast(C.byref(raw,4),C.POINTER(W.DWORD)).contents.value
                array=(ULONG_PTR*found).from_buffer(raw,8)
                return [int(x) for x in array]
            error=C.get_last_error()
            if error!=234:raise C.WinError(error,'Job member list')
            count*=2
        raise RuntimeError('Job membership exceeds qualified metadata envelope')

    def check_identity(self,pid,creation_time):
        if pid!=self.pid or abs(creation_time-self.creation_time)>.001:raise RuntimeError('Telemetry PID/create-time does not match owned process handle')
        return True

    def resume(self):
        if self._closed or self._thread_resumed:raise RuntimeError('No duplicate or closed resume')
        try:
            self._member()
            if self.query()['ActiveProcesses']!=1:raise RuntimeError('Unexpected suspended family state')
            if self._fault:self._fault('before_resume',self)
            previous=self._api.k.ResumeThread(self._pi.hThread)
            if previous!=1:
                if previous==0xffffffff:raise C.WinError(C.get_last_error(),'ResumeThread')
                raise RuntimeError('Unexpected initial suspend count')
            self._thread_resumed=True;self.stages.append('RESUMED_AFTER_OWNERSHIP')
            self._api.k.CloseHandle(self._pi.hThread);self._pi.hThread=None
        except BaseException:
            self.close();raise
        return self

    def root_poll(self):
        if self.root_returncode is not None:return self.root_returncode
        if not self._pi.hProcess:return None
        signaled=self._api.k.WaitForSingleObject(self._pi.hProcess,0)
        if signaled==0:
            code=W.DWORD();self._api.ok(self._api.k.GetExitCodeProcess(self._pi.hProcess,C.byref(code)),'Process exit code')
            self.root_returncode=int(code.value)
        elif signaled!=258:raise C.WinError(C.get_last_error(),'Process wait')
        return self.root_returncode

    def poll(self):
        root=self.root_poll()
        if root is not None and self.query()['ActiveProcesses']==0:self.returncode=root
        return self.returncode

    def wait(self,timeout=None):
        deadline=None if timeout is None else time.monotonic()+timeout
        while self.poll() is None:
            if deadline is not None and time.monotonic()>=deadline:raise subprocess.TimeoutExpired(self.args,timeout)
            time.sleep(.01)
        return self.returncode

    wait_tree=wait

    def terminate_tree(self,exit_code=0xE210,timeout=5):
        if self._closed:return self.cleanup_receipt
        self._api.ok(self._api.k.TerminateJobObject(self._job,exit_code),'Terminate owned job')
        return self.wait_tree(timeout)

    def terminate(self):return self.terminate_tree()
    def kill(self):return self.terminate_tree()

    def close(self):
        if self._closed:return self.cleanup_receipt
        receipt=dict(pid=self.pid,creation_time=self.creation_time,creation_time_100ns=self.creation_time_100ns,
                     resumed=self._thread_resumed,stages=list(self.stages),family_exit_verified=False)
        error=None
        try:
            if self._job:
                initial=self._raw_query();receipt['before_close']=initial
                if initial['ActiveProcesses']:
                    self._api.ok(self._api.k.TerminateJobObject(self._job,0xE210),'Terminate owned family for close')
                deadline=time.monotonic()+5
                while (final:=self._raw_query())['ActiveProcesses']:
                    if time.monotonic()>=deadline:raise RuntimeError('Owned job did not become empty')
                    time.sleep(.01)
                if self._pi.hProcess:
                    if self._api.k.WaitForSingleObject(self._pi.hProcess,5000)!=0:raise RuntimeError('Root process not signaled after empty job')
                    self.root_poll();self.returncode=self.root_returncode
                receipt.update(final_job_accounting=final,family_exit_verified=True)
        except BaseException as failure:
            error=failure;receipt['cleanup_error']=repr(failure)
        finally:
            # Last handle close is an independent kernel kill-on-close backstop.
            for name in ['hThread','hProcess']:
                handle=getattr(self._pi,name)
                if handle:self._api.k.CloseHandle(handle);setattr(self._pi,name,None)
            if self._job:self._api.k.CloseHandle(self._job);self._job=None
            self._release_setup()
            for stream in self._streams:
                try:stream.close()
                except OSError:pass
            self._closed=True;self.cleanup_receipt=receipt
        if error:raise error
        return receipt

    def __enter__(self):return self
    def __exit__(self,*unused):self.close()


def owned_popen(*args,**kwargs):
    """Explicit factory alias for runner integration; never auto-resumes."""
    return OwnedPopen(*args,**kwargs)
