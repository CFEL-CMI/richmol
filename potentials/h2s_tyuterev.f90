! Potential energy function for XY2-type triatomic molecule, adapted from TROVE pot_xy2.f90 module
! TODO: Need to add relevant publication refs, H2S works of Vladimir Tyuterev

subroutine h2s_tyuterev(npoints, r12, r32, alpha, f)
  implicit none

  integer, intent(in) :: npoints
  real(8), intent(in) :: r12(npoints), r32(npoints), alpha(npoints)
  real(8), intent(out) :: f(npoints)

  integer, parameter :: nparams = 102
  real(8) :: params(nparams)

  params = (/ &
         1.3359007d0,          &             !re12
         1.70400000d0,         &             !aa1
         92.265883d0,          &             !alphae
         0.80000000000000d+06, &             !b1
         0.80000000000000d+05, &             !b2
         0.13000000000000d+02, &             !g1
         0.55000000000000d+01, &             !g2
         0.00000000000000d+00, &             !f000
         0.25298724728304d+01, &             !f001
         0.76001446034650d+01, &             !f100
         0.19119869561968d+05, &             !f002
        -0.26337942369521d+04, &             !f101
        -0.34641383728936d+03, &             !f110
         0.37146640335080d+05, &             !f200
         0.10372456946123d+04, &             !f003
        -0.48366811961179d+04, &             !f102
         0.31178979423415d+04, &             !f111
        -0.12711176182111d+04, &             !f201
        -0.18726215609860d+03, &             !f210
        -0.11796148983875d+04, &             !f300
         0.47473763424544d+04, &             !f004
        -0.13610350468586d+04, &             !f103
         0.29144380025635d+04, &             !f112
        -0.50653357937930d+04, &             !f202
        -0.11097146714230d+04, &             !f211
         0.10375282984635d+03, &             !f220
         0.48047328722297d+04, &             !f301
        -0.14060018110572d+03, &             !f310
         0.22425353263397d+04, &             !f400
         0.17914500029660d+04, &             !f005
        -0.98184202293571d+03, &             !f104
         0.61294305063749d+02, &             !f113
        -0.27151696719548d+04, &             !f203
         0.26119013667838d+04, &             !f212
        -0.61703073530707d+04, &             !f221
        -0.48833736184025d+04, &             !f302
         0.10049289144454d+04, &             !f311
         0.28708061338852d+03, &             !f320
         0.67852017509792d+04, &             !f401
        -0.10489708631647d+04, &             !f410
         0.74727382498639d+03, &             !f500
         0.20496523413380d+04, &             !f006
         0.24807517966824d+03, &             !f105
        -0.19578362789532d+04, &             !f114
         0.70166633265091d+02, &             !f204
         0.51823676755422d+04, &             !f213
        -0.29736603824162d+04, &             !f222
        -0.39068887749886d+04, &             !f303
         0.22707809418082d+04, &             !f312
         0.23793551966211d+05, &             !f321
        -0.34932314882947d+04, &             !f330
         0.85923207392547d+03, &             !f402
        -0.11327987267957d+05, &             !f411
         0.37668571305335d+03, &             !f420
        -0.15953549707091d+05, &             !f501
         0.14890386471456d+04, &             !f510
         0.51484402685677d+03, &             !f600
        -0.54617014746870d+04, &             !f007
        -0.69943515992635d+02, &             !f106
         0.17427705522754d+04, &             !f115
         0.29147560120520d+03, &             !f205
        -0.33646004327917d+04, &             !f214
        -0.11202273117182d+04, &             !f223
        -0.38323335633188d+03, &             !f304
         0.26441151817245d+04, &             !f313
         0.19366049577246d+04, &             !f322
         0.38175718811604d+04, &             !f331
         0.30778367811357d+04, &             !f403
        -0.26353881747532d+04, &             !f412
         0.83999500575225d+04, &             !f421
        -0.52450951528119d+04, &             !f430
        -0.34729416630406d+03, &             !f502
        -0.50050684462558d+04, &             !f511
        -0.28211904320344d+04, &             !f520
        -0.10580696087439d+05, &             !f601
         0.53782996209764d+04, &             !f610
        -0.99734750908249d+03, &             !f700
        -0.74046834718425d+03, &             !f008
         0.00000000000000d+00, &             !f107
         0.00000000000000d+00, &             !f116
         0.00000000000000d+00, &             !f206
         0.00000000000000d+00, &             !f215
         0.00000000000000d+00, &             !f224
         0.00000000000000d+00, &             !f305
         0.00000000000000d+00, &             !f314
         0.00000000000000d+00, &             !f323
         0.00000000000000d+00, &             !f332
         0.00000000000000d+00, &             !f404
         0.00000000000000d+00, &             !f413
         0.00000000000000d+00, &             !f422
         0.00000000000000d+00, &             !f431
         0.00000000000000d+00, &             !f440
         0.00000000000000d+00, &             !f503
         0.00000000000000d+00, &             !f512
         0.00000000000000d+00, &             !f521
         0.00000000000000d+00, &             !f530
         0.00000000000000d+00, &             !f602
         0.00000000000000d+00, &             !f611
         0.00000000000000d+00, &             !f620
         0.00000000000000d+00, &             !f701
         0.00000000000000d+00, &             !f710
         0.00000000000000d+00 /)             !f800

  call poten_xy2_tyuterev(npoints, nparams, r12, r32, alpha, params, f)

end subroutine h2s_tyuterev


subroutine poten_xy2_tyuterev(npoints, nparams, r12, r32, alpha, params, f)
    implicit none
  
    integer, intent(in) :: nparams, npoints
    real(8), intent(in) :: r12(npoints), r32(npoints), alpha(npoints)
    real(8), intent(in) :: params(nparams)
    real(8), intent(out) :: f(npoints)
  
    real(8), dimension(npoints) :: v,v0,v1,v2,v3,v4,v5,v6,v7,v8,rhh,vhh,y1,y2,y3
    real(8) :: force(nparams-3)
    real(8) :: aa1,re12,alphae,g1,g2,b1,b2
    integer :: N 
  
    re12   = params(1)
    aa1    = params(2)
    alphae = params(3) * acos(-1.0d0)/180.0d0
  
    force(1:nparams-3) = params(4:nparams)
  
    b1   = force(1)
    b2   = force(2)
    g1   = force(3)
    g2   = force(4)
  
    rhh=sqrt(r12**2+r32**2-2.d0*r12*r32*cos(alpha))
    vhh=b1*exp(-g1*rhh)+b2*exp(-g2*rhh**2)
  
    y1 = 1.0d0-exp(-aa1*(r12-re12))
    y2 = 1.0d0-exp(-aa1*(r32-re12))
    y3 = cos(alpha)-cos(alphae)
  
    N = size(force)
  
    v4 = 0 ; v5 = 0 ; v6 = 0 ; v7 = 0 ; v8 = 0
  
    v0 = force(5)*y1**0*y2**0*y3**0
    v1 = force(6)*y1**0*y2**0*y3**1&
       + force(7)*y1**1*y2**0*y3**0&
       + force(7)*y1**0*y2**1*y3**0
    v2 = force(8)*y1**0*y2**0*y3**2&
       + force(9)*y1**1*y2**0*y3**1&
       + force(9)*y1**0*y2**1*y3**1&
       + force(10)*y1**1*y2**1*y3**0&
       + force(11)*y1**2*y2**0*y3**0&
       + force(11)*y1**0*y2**2*y3**0
    v3 = force(12)*y1**0*y2**0*y3**3&
       + force(13)*y1**1*y2**0*y3**2&
       + force(13)*y1**0*y2**1*y3**2&
       + force(14)*y1**1*y2**1*y3**1&
       + force(15)*y1**2*y2**0*y3**1&
       + force(15)*y1**0*y2**2*y3**1&
       + force(16)*y1**2*y2**1*y3**0&
       + force(16)*y1**1*y2**2*y3**0&
       + force(17)*y1**3*y2**0*y3**0&
       + force(17)*y1**0*y2**3*y3**0
  
    if (N>18) then
      v4 = force(18)*y1**0*y2**0*y3**4&
      + force(19)*y1**1*y2**0*y3**3&
      + force(19)*y1**0*y2**1*y3**3&
      + force(20)*y1**1*y2**1*y3**2&
      + force(21)*y1**2*y2**0*y3**2&
      + force(21)*y1**0*y2**2*y3**2&
      + force(22)*y1**2*y2**1*y3**1&
      + force(22)*y1**1*y2**2*y3**1&
      + force(23)*y1**2*y2**2*y3**0&
      + force(24)*y1**3*y2**0*y3**1&
      + force(24)*y1**0*y2**3*y3**1&
      + force(25)*y1**3*y2**1*y3**0&
      + force(25)*y1**1*y2**3*y3**0&
      + force(26)*y1**4*y2**0*y3**0&
      + force(26)*y1**0*y2**4*y3**0
  endif
  
  if (N>26) then
    v5 = force(27)*y1**0*y2**0*y3**5&
      + force(28)*y1**1*y2**0*y3**4&
      + force(28)*y1**0*y2**1*y3**4&
      + force(29)*y1**1*y2**1*y3**3&
      + force(30)*y1**2*y2**0*y3**3&
      + force(30)*y1**0*y2**2*y3**3&
      + force(31)*y1**2*y2**1*y3**2&
      + force(31)*y1**1*y2**2*y3**2&
      + force(32)*y1**2*y2**2*y3**1&
      + force(33)*y1**3*y2**0*y3**2&
      + force(33)*y1**0*y2**3*y3**2&
      + force(34)*y1**3*y2**1*y3**1&
      + force(34)*y1**1*y2**3*y3**1&
      + force(35)*y1**3*y2**2*y3**0&
      + force(35)*y1**2*y2**3*y3**0&
      + force(36)*y1**4*y2**0*y3**1&
      + force(36)*y1**0*y2**4*y3**1&
      + force(37)*y1**4*y2**1*y3**0&
      + force(37)*y1**1*y2**4*y3**0&
      + force(38)*y1**5*y2**0*y3**0&
      + force(38)*y1**0*y2**5*y3**0
  endif
  
  if (N>38) then
    v6 = force(39)*y1**0*y2**0*y3**6&
      + force(40)*y1**1*y2**0*y3**5&
      + force(40)*y1**0*y2**1*y3**5&
      + force(41)*y1**1*y2**1*y3**4&
      + force(42)*y1**2*y2**0*y3**4&
      + force(42)*y1**0*y2**2*y3**4&
      + force(43)*y1**2*y2**1*y3**3&
      + force(43)*y1**1*y2**2*y3**3&
      + force(44)*y1**2*y2**2*y3**2&
      + force(45)*y1**3*y2**0*y3**3&
      + force(45)*y1**0*y2**3*y3**3&
      + force(46)*y1**3*y2**1*y3**2&
      + force(46)*y1**1*y2**3*y3**2&
      + force(47)*y1**3*y2**2*y3**1&
      + force(47)*y1**2*y2**3*y3**1&
      + force(48)*y1**3*y2**3*y3**0&
      + force(49)*y1**4*y2**0*y3**2&
      + force(49)*y1**0*y2**4*y3**2&
      + force(50)*y1**4*y2**1*y3**1&
      + force(50)*y1**1*y2**4*y3**1&
      + force(51)*y1**4*y2**2*y3**0&
      + force(51)*y1**2*y2**4*y3**0&
      + force(52)*y1**5*y2**0*y3**1&
      + force(52)*y1**0*y2**5*y3**1&
      + force(53)*y1**5*y2**1*y3**0&
      + force(53)*y1**1*y2**5*y3**0&
      + force(54)*y1**6*y2**0*y3**0&
      + force(54)*y1**0*y2**6*y3**0
  endif
  
  if (N>54) then
    v7 = force(55)*y1**0*y2**0*y3**7&
      + force(56)*y1**1*y2**0*y3**6&
      + force(56)*y1**0*y2**1*y3**6&
      + force(57)*y1**1*y2**1*y3**5&
      + force(58)*y1**2*y2**0*y3**5&
      + force(58)*y1**0*y2**2*y3**5&
      + force(59)*y1**2*y2**1*y3**4&
      + force(59)*y1**1*y2**2*y3**4&
      + force(60)*y1**2*y2**2*y3**3&
      + force(61)*y1**3*y2**0*y3**4&
      + force(61)*y1**0*y2**3*y3**4&
      + force(62)*y1**3*y2**1*y3**3&
      + force(62)*y1**1*y2**3*y3**3&
      + force(63)*y1**3*y2**2*y3**2&
      + force(63)*y1**2*y2**3*y3**2&
      + force(64)*y1**3*y2**3*y3**1&
      + force(65)*y1**4*y2**0*y3**3&
      + force(65)*y1**0*y2**4*y3**3&
      + force(66)*y1**4*y2**1*y3**2&
      + force(66)*y1**1*y2**4*y3**2&
      + force(67)*y1**4*y2**2*y3**1&
      + force(67)*y1**2*y2**4*y3**1&
      + force(68)*y1**4*y2**3*y3**0&
      + force(68)*y1**3*y2**4*y3**0&
      + force(69)*y1**5*y2**0*y3**2&
      + force(69)*y1**0*y2**5*y3**2&
      + force(70)*y1**5*y2**1*y3**1&
      + force(70)*y1**1*y2**5*y3**1&
      + force(71)*y1**5*y2**2*y3**0&
      + force(71)*y1**2*y2**5*y3**0&
      + force(72)*y1**6*y2**0*y3**1&
      + force(72)*y1**0*y2**6*y3**1&
      + force(73)*y1**6*y2**1*y3**0&
      + force(73)*y1**1*y2**6*y3**0&
      + force(74)*y1**7*y2**0*y3**0&
      + force(74)*y1**0*y2**7*y3**0
  endif
  
  if (N>74) then
    v8 = force(75)*y1**0*y2**0*y3**8&
      + force(76)*y1**1*y2**0*y3**7&
      + force(76)*y1**0*y2**1*y3**7&
      + force(77)*y1**1*y2**1*y3**6&
      + force(78)*y1**2*y2**0*y3**6&
      + force(78)*y1**0*y2**2*y3**6&
      + force(79)*y1**2*y2**1*y3**5&
      + force(79)*y1**1*y2**2*y3**5&
      + force(80)*y1**2*y2**2*y3**4&
      + force(81)*y1**3*y2**0*y3**5&
      + force(81)*y1**0*y2**3*y3**5&
      + force(82)*y1**3*y2**1*y3**4&
      + force(82)*y1**1*y2**3*y3**4&
      + force(83)*y1**3*y2**2*y3**3&
      + force(83)*y1**2*y2**3*y3**3&
      + force(84)*y1**3*y2**3*y3**2&
      + force(85)*y1**4*y2**0*y3**4&
      + force(85)*y1**0*y2**4*y3**4&
      + force(86)*y1**4*y2**1*y3**3&
      + force(86)*y1**1*y2**4*y3**3&
      + force(87)*y1**4*y2**2*y3**2&
      + force(87)*y1**2*y2**4*y3**2&
      + force(88)*y1**4*y2**3*y3**1&
      + force(88)*y1**3*y2**4*y3**1&
      + force(89)*y1**4*y2**4*y3**0&
      + force(90)*y1**5*y2**0*y3**3&
      + force(90)*y1**0*y2**5*y3**3&
      + force(91)*y1**5*y2**1*y3**2&
      + force(91)*y1**1*y2**5*y3**2&
      + force(92)*y1**5*y2**2*y3**1&
      + force(92)*y1**2*y2**5*y3**1&
      + force(93)*y1**5*y2**3*y3**0&
      + force(93)*y1**3*y2**5*y3**0&
      + force(94)*y1**6*y2**0*y3**2&
      + force(94)*y1**0*y2**6*y3**2&
      + force(95)*y1**6*y2**1*y3**1&
      + force(95)*y1**1*y2**6*y3**1&
      + force(96)*y1**6*y2**2*y3**0&
      + force(96)*y1**2*y2**6*y3**0&
      + force(97)*y1**7*y2**0*y3**1&
      + force(97)*y1**0*y2**7*y3**1&
      + force(98)*y1**7*y2**1*y3**0&
      + force(98)*y1**1*y2**7*y3**0&
      + force(99)*y1**8*y2**0*y3**0&
      + force(99)*y1**0*y2**8*y3**0
  endif
  
  f=v0+v1+v2+v3+v4+v5+v6+v7+v8+vhh
  
  end subroutine poten_xy2_tyuterev