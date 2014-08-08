
PRO aia_prep, input1, input2, oindex, odata, index_ref=index_ref, data_ref=data_ref, $
  use_ref=use_ref, infil=infil, cutout=cutout, $
  use_hdr_pnt=use_hdr_pnt, mpt_rec_num=mpt_rec_num, t_ref=t_ref, $
  no_uncomp_delete=no_uncomp_delete, nearest=nearest, interp=interp, cubic=cubic, $
  normalize=normalize, $
  do_write_fits=do_write_fits, outdir=outdir, outfile=outfile, _extra=_extra, $
  verbose=verbose, quiet=quiet, run_time=run_time, $
  progver_main=progver_main, prognam_main=prognam_main, $
  qstop=qstop, $
  $
  use_test_image=use_test_image, use_pnt_file=use_pnt_file, $
  not_use_ssw=not_use_ssw, $
  scale_fac=scale_fac, sign_mag=sign_mag, sign_x0=sign_x0, sign_y0=sign_y0, sign_angle=sign_angle

;+
; NAME:
;   AIA_PREP
; PURPOSE:
;   Perform image registration (rotation, translation, scaling) of Level 1 AIA images, and update
;   the header information.
; CATEGORY:
;   Image alignment
; SAMPLE CALLS:
;   Inputing infil (in this case iindex and idata are returned with 
;     IDL> AIA_PREP, infil, [0,1,2], oindex, odata
;   Inputing iindex and idata: 
;     IDL> AIA_PREP, iindex, idata, oindex, odata
;   Same, but align to coordinates specified by INDEX_REF
;     IDL> AIA_PREP, infil, [0,1,2], oindex, odata, index_ref=index_ref
;   This time, align to first image (pointing, scale, roll):
;     IDL> AIA_PREP, infil, [0,1,2], oindex, odata, /cutout
;   Write our registered date, header to FITS file (using parameter 'OUTDIR', default is
;     current directory):
;     IDL> AIA_PREP, iindex, idata, /do_write_fits, outdir=outdir
; INPUTS:
;   There are 2 basic usages for inputing the image and header data into AIA_PREP:
;   Case 1: References FITS file name on disk:
;           input1 - String array list of AIA FITS files
;           input2 - List of indices of FITS files to read 
;   Case 2. References index structure and data array in memory
;           (index, data already read from FITS file using, for example, READ_SDO.PRO):
;           input1 - index structure
;           input2 - data array
; OUTPUTS (OPTIONAL):
;   oindex - The updated index structure of the input image
;   odata - Registered output image.
; KEYWORDS:
;   USE_REF - If set then align all images to a reference index (if INDEX_REF is not supplied then
;             use the first index of the array as the reference index).
;             NOTE - If USE_REF is not set and if INDEX_REF is not supplied, then all images will
;             be aligned to sun center.
;   CUTOUT - Same effect as USE_REF above.
;   INDEX_REF - Reference index for alignment coordinates.
;   DO_WRITE_FITS - If set, write the registered image and updated header structure to disk
;   NEAREST - If set, use nearest neighbor interpolatipon
;   INTERP - If set, use bilinear interpolation
;   CUBIC - If set, use cubic convolution interpolation ith the specified value (in the range [-1,0]
;           as the interpolation parameter.  Cubic interpolation with this parameter equal -0.5
;           is the default.
;   USE_HDR_PNT - If set, force use of header values for roll and scale.  By default, the latest
;                 MPT record is accesses for these.
;   MPT_REC_NUM - If passed, use these MPT record numbers for roll and scale rather than either
;                 the header values or the nearest in time MPT record numbers.  Number of elements
;                 of MPT_REC_NUM must equal either 1 or the number of images (or FITS files) passed.
;                 If one record number passed then that one is used for every image.
;   T_REF - If passed, use the MPT record with the nearest earlier MPO_REC.DATE to T_REF for
;     which the INDEX.DATE_OBS tag falls within the range [MPO_REC.T_START, MPO_REC.T_END].
;   NORMALIZE - If set, and image is AIA, then do exposure normalization.
; HISTORY:
;   2010 (circa), Created ab initio - GLS (slater@lmsal.com)
;   2010-12-07 - GLS - Corrected call to break_file - GLS
;   2011-02-10 - GLS - 1. Corrected sign error on roll (Thanks to Ralph Seguin)
;                      2. Corrections to tags CRPIX(1,2), CDELT(1,2, and CROT2A were not being 
;                         propagated to output header structure (OINDEX).  This was fixed
;                         (Thanks to Benjamin Mampaey)
;   2011-02-28 - GLS - 1. Added missing half pixel to output CRPIX1/2
;                      2. Defined LVL_NUM keyword to be 1.5 (should it be 1.51 to differentiate from
;                         real time?)
;                      3. Added _EXTRA in call for keyword inheritance
;                      4. Made UNCOMP_DELETE the default in call to READ_SDO.
;   2011-03-02 - GLS - 1. Corrected references to NAXIS1/2 in case of compressed file headers
;                      2. Changed default interpolation for ROT function from nearest neighbor to
;                         damped cubic.
;   2011-04-07 - GLS - Corrected 1 pixel error (both axes) in pivot point for ROT function due to
;                      discrepancy between FITS standard pixel numbering for CRPIX1,2 (starting from 1)
;                      and IDL array index referencing (starting from 0).  Thanks to Alberto Vasquez.
;   2011-04-08 - GLS - Added '/pivot' to ROT call for 'cutout' images.  Thanks to Marc DeRosa.
;   2011-04-20 - GLS - Checked for existence of tags 'RSUN_OBS', 'RSUN', and 'LVL_NUM' before attempting
;                      to update them, in order for code to work on HMI images.
;   2011-05-10 - GLS - Removed incorrect logic to correct for binning.
;   2011-05-18 - GLS - Corrected multiple errors in naming convention for output FITS files in HMI case
;   2011-05-19 - GLS - Added updating of header tags:
;                        DATAMIN, DATAMAX, DATAMEDN, DATAMEAN, DATARMS, DATASKEW, DATAKURT
;   2011-05-20 - GLS - Changed from inserting OINDEX0 into pre-defined OINDEX array back to concatenation
;                        cause I couldn't figger out a bonehead error.
;   2011-06-08 - GLS - Corrected error in usage of OUTFILE parameter.
;   2011-07-13 - GLS - Corrected errors in processing of 'cutout' images.
;   2011-12-19 - GLS - Corrected CRPIX error in HMI cutouts.
;   2011-12-20 - GLS - Further correction of CRPIX errors.
;   2012-07-26 - GLS - Inserted special handling of single file case to remove redundant read_sdo call
;                      identified by Marc DeRosa.
;   2012-08-12 - GLS - 1. Added Venus Transit-derived CDELT and CROTA corrections for older files.
;                      2. Added keyword NO_ASTROMETRY to inhibit application of 2012 Venus transit (and
;                         perhaps other event) pointing updates to pre_transit/event files.
;                      3. Replaced keyword CUTOUT with keyword CUTOUT_CENTER and made it function as
;                         keyword USE_REF.
;   2012-09-10 - GLS - Changed the deafult for handling scale/roll updates.  Now by default the latest
;                      MPT record for every image header is read and used for roll and scale values.
;                      User may force use of header values by setting USE_HDR_PNT flag.  Alternatively,
;                      user may pass MPT record number to use via MPT_REC_NUM keyword (see above).
;                      The NO_ASTROMETRY keyword is eliminated.
;   2012-10-30 - GLS - Major re-design.  Broke into functional pieces leaving AIA_PREP.PRO as just the
;                      front end.
;   2012-11-15 - GLS - Added missing default for N_NOT_EXIST_MAX , and modified call to AIA_REG .
;                      Thanks to Giuliana de Toma, Trae Winter, David Shelton for bug report.
;   2013-01-07 - GLS - Added T_REF and MPT_REC_NUM keywords.
;   2013-04-20 - GLS - Added NORMALIZE keyword.
;   2013-05-20 - GLS - Corrected bug in handling of cutouts.
;   2013-06-27 - GLS - Further debugs...(sigh)...
;   2013-07-22 - GLS, Alessandro Cilla - Corrected bugs associated with inputing files instead of
;                                        INDEX and DATA variables.
;-

; Define prognam, progver variables
prognam = 'AIA_PREP.PRO'
prognam_main = prognam

;progver = 'V4.00' ; 2011-02-10 (GLS)
;progver = 'V4.01' ; 2011-03-01 (GLS)
;progver = 'V4.02' ; 2011-03-02 (GLS)
;progver = 'V4.03' ; 2011-04-06 (GLS)
;progver = 'V4.04' ; 2011-04-07 (GLS)
;progver = 'V4.05' ; 2011-04-20 (GLS)
;progver = 'V4.06' ; 2011-05-10 (GLS)
;progver = 'V4.07' ; 2011-05-18 (GLS)
;progver = 'V4.08' ; 2011-05-18 (GLS)
;progver = 'V4.09' ; 2011-05-20 (GLS)
;progver = 'V4.10' ; 2011-06-08 (GLS)
;progver = 'V4.11' ; 2011-06-13 (GLS)
;progver = 'V4.12' ; 2011-12-19 (GLS)
;progver = 'V4.13' ; 2011-12-22 (GLS)
;progver = 'V4.14' ; 2012-08    (GLS)
;progver = 'V4.15' ; 2012-09-10 (GLS)
;progver = 'V4.16' ; 2012-09-10 (GLS)
;progver = 'V4.17' ; 2012-09-10 (GLS)
;progver = 'V5.00' ; 2013-05-20 (GLS)
progver = 'V5.10' ; 2013-06-27 (GLS)
progver_main = progver

verbose = keyword_set(verbose)
if (verbose eq 1) then loud = 1
if verbose then print, 'Running ', prognam, ' ', progver

if not exist(n_not_exist_max) then n_not_exist_max = 1

; Start runtime clock running
t0 = systime(1)
t1 = t0	; Keep track of running time

; Set miscellaneous variables, flags, etc:
use_ref = ( keyword_set(use_ref) or keyword_set(cutout) or exist(index_ref) )
cutout  = ( keyword_set(use_ref) or keyword_set(cutout) or exist(index_ref) )

hmi_content_value = $
  ['dopplergram', 'magnetogram', 'level 1p image', 'linewidth', 'linedepth', 'contiuum intensity']
hmi_outfil_suffix = $
  ['dop','mag','img','wid','dep','cont']

; -----------------------------------------------------------
; Beginning of input identifcation and verification section
; -----------------------------------------------------------

; Case of insufficient inputs:
if n_params() lt 2 then begin
  err_mess = ' Input error: Minimum input parameters is 2 (file list and index list).  Returning.'
  print, ' Input error: Minimum input parameters is 2 (file list and index list).  Returning.'
  return
endif

; Case of file list/index list inputs:
if size(input1, /tname) eq 'STRING' then begin
  input_mode = 'file_list'
  input_err = 0

  ss_infil = input2
  if ss_infil[0] eq -1 then ss_infil = indgen(n_elements(input1))
  ss_not_exist = where(file_exist(input1[ss_infil]) ne 1, n_not_exist)
  if n_not_exist gt n_not_exist_max then begin
    err_mess = ' Input error: Not all files in file list found.  Returning.' 
    print, ' Input error: Not all files in file list found.  Returning.'
  endif

  n_img = n_elements(ss_infil)
  infil_arr = input1[ss_infil]

; If fourth parameter passed then read first file to create template data array:
;  if n_params() ge 4 then begin
;    read_sdo, infil_arr[0], iindex0, idata0, uncomp_delete=uncomp_delete, _extra=_extra
;    data_type = size(idata0, /type)
;    data_dim = size(idata0, /dim)
;    data_ndim = size(idata0, /n_dim)
;    odata = make_array([data_dim[0], data_dim[1], n_img], type=data_type)
;  endif

; If first header not yet read then read it now (for tag access):
  if not exist(iindex0) then read_sdo, infil_arr[0], iindex0, /nodata, _extra=_extra
endif

; Case of index/data inputs:
if size(input1, /tname) eq 'STRUCT' then begin
  input_mode = 'index_data'
  iindex0 = input1[0]
  n_img = n_elements(input1)
;  data_type = size(input2, /type)
;  data_dim = size(input2, /dim)
;  data_ndim = size(input2, /n_dim)

; Create empty array for odata, if fourth parameter passed:
;  if n_params() ge 4 then odata = make_array([data_dim[0], data_dim[1], n_img], type=data_type)
endif

;instr_prefix = strupcase(strmid(iindex0.instrume,0,3))

; -----------------------------------------------------
; End of input identifcation and verification section
; -----------------------------------------------------

; Now call the front end routine appropriate to the instrument:

;case instr_prefix of
;  'AIA': begin
    case n_params() of
      4: aia_reg, input1, input2, oindex, odata, input_mode=input_mode, $
           index_ref=index_ref, data_ref=data_ref, use_ref=use_ref, $
           infil=infil, cutout=cutout, nearest=nearest, interp=interp, cubic=cubic, $
           use_hdr_pnt=use_hdr_pnt, mpt_rec_num=mpt_rec_num, t_ref=t_ref, $
           no_uncomp_delete=no_uncomp_delete, $
           normalize=normalize, $
           do_write_fits=do_write_fits, outdir=outdir, outfile=outfile, _extra=_extra, $
           verbose=verbose, run_time=run_time, $
           progver_main=progver_main, prognam_main=prognam_main, $
           qstop=qstop
;      3: aia_reg, input1, input2, oindex, input_mode=input_mode, $
;           index_ref=index_ref, data_ref=data_ref, use_ref=use_ref, $
;           infil=infil, cutout=cutout, nearest=nearest, interp=interp, cubic=cubic, $
;           use_hdr_pnt=use_hdr_pnt, mpt_rec_num=mpt_rec_num, no_uncomp_delete=no_uncomp_delete, $
;           do_write_fits=do_write_fits, outdir=outdir, outfile=outfile, _extra=_extra, $
;           verbose=verbose, run_time=run_time, $
;           progver_main=progver_main, prognam_main=prognam_main, $
;           qstop=qstop
      else: aia_reg, input1, input2, input_mode=input_mode, $
              index_ref=index_ref, data_ref=data_ref, use_ref=use_ref, $
              infil=infil, cutout=cutout, nearest=nearest, interp=interp, cubic=cubic, $
              use_hdr_pnt=use_hdr_pnt, mpt_rec_num=mpt_rec_num, t_ref=t_ref, $
              no_uncomp_delete=no_uncomp_delete, $
              normalize=normalize, $
              do_write_fits=do_write_fits, outdir=outdir, outfile=outfile, _extra=_extra, $
              verbose=verbose, run_time=run_time, $
              progver_main=progver_main, prognam_main=prognam_main, $
              qstop=qstop

;            aia_reg, input1, input2, oindex, odata, input_mode=input_mode, $
;              index_ref=index_ref, use_ref=use_ref, $
;              infil=infil, cutout=cutout, nearest=nearest, interp=interp, cubic=cubic, $
;              use_hdr_pnt=use_hdr_pnt, mpt_rec_num=mpt_rec_num, t_ref=t_ref, $
;              no_uncomp_delete=no_uncomp_delete, $
;              normalize=normalize, $
;              do_write_fits=do_write_fits, outdir=outdir, outfile=outfile, _extra=_extra, $
;              verbose=verbose, run_time=run_time, $
;              progver_main=progver_main, prognam_main=prognam_main, $
;              qstop=qstop
    endcase
    end

;  'HMI': begin
;    hmi_prep, input1, input2, oindex, odata, index_ref=index_ref, data_ref=data_ref, $
;      use_ref=use_ref, $
;      infil=infil, use_test_image=use_test_image, cutout=cutout, $
;      no_astrometry=no_astrometry, $
;      use_pnt_file=use_pnt_file, not_use_ssw=not_use_ssw, no_uncomp_delete=no_uncomp_delete, $
;      nearest=nearest, interp=interp, cubic=cubic, $
;      do_write_fits=do_write_fits, outdir=outdir, outfile=outfile, scale_fac=scale_fac, _extra=_extra, $
;      sign_mag=sign_mag, sign_x0=sign_x0, sign_y0=sign_y0, sign_angle=sign_angle, $
;      qstop=qstop, quiet=quiet, verbose=verbose, run_time=run_time, progver=progver, prognam=prognam
;    end
;endcase

if keyword_set(qstop) then stop,' Stopping on request.'

end
